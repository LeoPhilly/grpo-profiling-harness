"""Mac-side tests for the autopsy's evidence-shaping logic (the bucketing —
a wrong bucket misleads the diagnosis). The GPU parts only run on the box."""

import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parent.parent / "scripts" / "identity_autopsy.py"
_spec = importlib.util.spec_from_file_location("identity_autopsy", _path)
identity_autopsy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(identity_autopsy)

position_bucket = identity_autopsy.position_bucket
autopsy_out_dir = identity_autopsy.autopsy_out_dir


def test_out_dir_fresh_vs_checkpoint():
    root = Path("/results")
    # Fresh-weights run: unchanged v2 baseline dir.
    assert autopsy_out_dir(root, None) == root / "identity_autopsy" / "v2"
    assert autopsy_out_dir(root, "") == root / "identity_autopsy" / "v2"
    # --checkpoint run: isolated subdir from the checkpoint's basename, so it
    # can't overwrite the fresh-weights baseline.
    ckpt = "results/checkpoints/r2-g8-cache-on-1.5b-profile"
    assert autopsy_out_dir(root, ckpt) == (
        root / "identity_autopsy" / "v2" / "checkpoint-r2-g8-cache-on-1.5b-profile"
    )
    # Trailing slash must still yield the dir name (os.path.basename would
    # return "" here; Path(...).name does not).
    assert autopsy_out_dir(root, "a/b/r2-g8/").name == "checkpoint-r2-g8"


def test_absolute_bins_with_boundary_rows():
    # L=300: final and last-3 take priority; everything else bins by
    # ABSOLUTE position. Bin boundaries are half-open: 63 in 0-64, 64 not.
    assert position_bucket(299, 300) == "final"
    assert [position_bucket(i, 300) for i in (296, 297, 298)] == ["last3"] * 3
    assert position_bucket(0, 300) == "0-64"
    assert position_bucket(63, 300) == "0-64"
    assert position_bucket(64, 300) == "64-128"
    assert position_bucket(127, 300) == "64-128"
    assert position_bucket(128, 300) == "128-256"
    assert position_bucket(255, 300) == "128-256"
    assert position_bucket(256, 300) == "256-512"


def test_buckets_are_disjoint_and_total_at_edge_lengths():
    for length in (1, 2, 3, 4, 5, 65, 130):
        buckets = [position_bucket(i, length) for i in range(length)]
        assert all(b in identity_autopsy.BUCKETS for b in buckets)
    # Priority at the edges: a 1-token completion is its own final token,
    # and in a 2-token one position 0 is within the last-3-before-final
    # window, so the boundary row claims it before any absolute bin.
    assert position_bucket(0, 1) == "final"
    assert [position_bucket(i, 2) for i in range(2)] == ["last3", "final"]
    assert [position_bucket(i, 5) for i in range(5)] == [
        "0-64", "last3", "last3", "last3", "final",
    ]
    # A deep position past every bin hits the safety fallback.
    assert position_bucket(600, 1000) == "512+"
