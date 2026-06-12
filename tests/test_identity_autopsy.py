"""Mac-side tests for the autopsy's evidence-shaping logic (the bucketing —
a wrong bucket misleads the diagnosis). The GPU parts only run on the box."""

import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parent.parent / "scripts" / "identity_autopsy.py"
_spec = importlib.util.spec_from_file_location("identity_autopsy", _path)
identity_autopsy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(identity_autopsy)

position_bucket = identity_autopsy.position_bucket


def test_buckets_for_normal_length():
    # L=10: 0=first, 9=final, 6/7/8=last3 (the three before final), rest middle.
    got = [position_bucket(i, 10) for i in range(10)]
    assert got == [
        "first", "middle", "middle", "middle", "middle", "middle",
        "last3", "last3", "last3", "final",
    ]


def test_buckets_are_disjoint_and_total_at_edge_lengths():
    for length in (1, 2, 3, 4, 5):
        buckets = [position_bucket(i, length) for i in range(length)]
        assert all(b in identity_autopsy.BUCKETS for b in buckets)
    # Priority at the edges: a 1-token completion is its own final token.
    assert position_bucket(0, 1) == "final"
    assert [position_bucket(i, 2) for i in range(2)] == ["first", "final"]
    assert [position_bucket(i, 5) for i in range(5)] == [
        "first", "last3", "last3", "last3", "final",
    ]
