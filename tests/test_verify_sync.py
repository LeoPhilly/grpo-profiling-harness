"""Mac-side tests for verify_sync's name-mapping/fusion logic — the subtle
part: a wrong fused slice would compare q rows against k rows and either
fake a failure or mask a real one. GPU stages only run on the box."""

import importlib.util
from pathlib import Path

import torch

_path = Path(__file__).resolve().parent.parent / "scripts" / "verify_sync.py"
_spec = importlib.util.spec_from_file_location("verify_sync", _path)
verify_sync = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify_sync)

locate = verify_sync.locate


def _synthetic():
    """Trainer q/k/v + gate/up vs an engine with fused tensors, with
    distinctive values per part so slice mistakes can't pass."""
    q = torch.full((4, 8), 1.0)
    k = torch.full((2, 8), 2.0)
    v = torch.full((2, 8), 3.0)
    gate = torch.full((6, 8), 4.0)
    up = torch.full((6, 8), 5.0)
    trainer = {
        "model.layers.0.self_attn.q_proj.weight": q,
        "model.layers.0.self_attn.k_proj.weight": k,
        "model.layers.0.self_attn.v_proj.weight": v,
        "model.layers.0.mlp.gate_proj.weight": gate,
        "model.layers.0.mlp.up_proj.weight": up,
        "model.embed_tokens.weight": torch.full((10, 8), 6.0),
    }
    engine = {
        "model.layers.0.self_attn.qkv_proj.weight": torch.cat([q, k, v], dim=0),
        "model.layers.0.mlp.gate_up_proj.weight": torch.cat([gate, up], dim=0),
        "model.embed_tokens.weight": trainer["model.embed_tokens.weight"].clone(),
    }
    return trainer, engine


def test_direct_and_fused_slices_recovered_exactly():
    trainer, engine = _synthetic()
    for name, tensor in trainer.items():
        hit = locate(name, trainer, engine)
        assert hit is not None, name
        engine_tensor, engine_name, how = hit
        assert torch.equal(engine_tensor, tensor), (name, how)
    # Slice arithmetic specifically: v sits after q (4 rows) + k (2 rows).
    _, _, how = locate("model.layers.0.self_attn.v_proj.weight", trainer, engine)
    assert how == "fused[6:8]"
    _, _, how = locate("model.layers.0.mlp.up_proj.weight", trainer, engine)
    assert how == "fused[6:12]"


def test_prefix_variants_and_unmatched():
    trainer, engine = _synthetic()
    # Engine without the "model." prefix still matches.
    stripped = {k[len("model."):]: v for k, v in engine.items()}
    hit = locate("model.embed_tokens.weight", trainer, stripped)
    assert hit is not None and hit[2] == "direct"
    # A name with no direct or fused counterpart returns None (reported,
    # never silently skipped, by the caller).
    assert locate("model.layers.0.self_attn.o_proj.weight", trainer, engine) is None
