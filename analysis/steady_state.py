#!/usr/bin/env python
"""Citable steady-state aggregates from a wandb run window.

Read-only analysis over logged history — touches no training code. The phase
table is only as trustworthy as standing check #2: a >5% mean |timing
residual| in the window prints a loud caveat.
"""

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

TIME_PREFIX = "time/"
WALL = "time/wall_clock"
TPS = "time/tokens_per_sec_generate"
RESIDUAL = "check/timing_residual_frac"
SCALARS = (TPS, "train/reward_mean", "train/format_rate")
# Runs after the flat phase decomposition log forward/loss_compute/backward
# separately; this derived row keeps them comparable with r0's logged
# time/forward_loss. NOTE: r0's aggregate also contained the identity
# computation (now its own identity_check phase), so the comparison is
# ~ms generous to new runs.
FORWARD_LOSS = "time/forward_loss"
FORWARD_LOSS_PARTS = ("time/forward", "time/loss_compute", "time/backward")
FORWARD_LOSS_DERIVED = "forward_loss (=forward+loss_compute+backward)"


def percentile(sorted_vals, q):
    """Nearest-rank-ish percentile on a pre-sorted list."""
    return sorted_vals[int(q * (len(sorted_vals) - 1))]


def _window(rows, start_step, end_step):
    """Rows whose _step is in [start_step, end_step)."""
    return [
        r
        for r in rows
        if isinstance(r.get("_step"), (int, float))
        and start_step <= r["_step"] < end_step
    ]


def _last_n(rows, n):
    """The n highest-_step rows (the run's tail) — NOT the last n of any
    window. Lets end-of-run drift show against the steady baseline."""
    valid = sorted(
        (r for r in rows if isinstance(r.get("_step"), (int, float))),
        key=lambda r: r["_step"],
    )
    return valid[-n:]


def _collect_series(window):
    """{key: [numeric values...]} over a list of history rows; non-numeric
    values (e.g. string notes) are skipped, so older runs missing a key
    simply produce no series for it."""
    series = {}
    for row in window:
        for key, val in row.items():
            if isinstance(val, (int, float)):
                series.setdefault(key, []).append(val)
    return series


def _bundle(vals):
    """Full stat bundle for a numeric series, or None if absent/empty (->
    'n/a' downstream, never a crash)."""
    if not vals:
        return None
    s = sorted(vals)
    return {
        "mean": statistics.fmean(s),
        "std": statistics.stdev(s) if len(s) > 1 else 0.0,
        "min": s[0],
        "max": s[-1],
        "p10": percentile(s, 0.1),
        "p90": percentile(s, 0.9),
        "n": len(s),
    }


def _fmt(v):
    """Magnitude-aware formatting: tiny values (identity, residual) keep 6
    decimals; mid values 4; large (token counts) 1. None -> 'n/a'."""
    if v is None:
        return "n/a"
    a = abs(v)
    if a != 0 and a < 0.01:
        return f"{v:.6f}"
    if a < 100:
        return f"{v:.4f}"
    return f"{v:.1f}"


# Section specs: (label, key, window, stats_to_surface). window is "steady"
# or "last_20". Residual and the section-4 deltas are built specially below.
SECTION2_SPECS = [
    ("straggler/p99_p50_ratio", "straggler/p99_p50_ratio", "steady",
     ("mean", "std", "p90", "max")),
    ("straggler/completion_len_max", "straggler/completion_len_max", "steady",
     ("mean", "max")),
    ("straggler/completion_len_median", "straggler/completion_len_median",
     "steady", ("mean", "std")),
    ("train/truncated_frac", "train/truncated_frac", "steady", ("mean",)),
    ("train/completion_tokens", "train/completion_tokens", "steady", ("mean",)),
]
SECTION3_IDENTITY_SPECS = [
    ("check/logprob_identity [steady]", "check/logprob_identity", "steady",
     ("mean",)),
    ("check/logprob_identity [last_20]", "check/logprob_identity", "last_20",
     ("mean",)),
    ("check/logprob_identity_min", "check/logprob_identity_min", "steady",
     ("min",)),
    ("check/logprob_identity_max", "check/logprob_identity_max", "steady",
     ("max",)),
]
STAT_COLS = ("mean", "std", "min", "max", "p10", "p90")
GPU_METRICS = (("GPU Utilization", "gpu"), ("GPU Time Accessing Memory", "memory"))


def aggregate_window(rows, start_step, end_step):
    """Pure aggregation over history rows (dicts with _step). Returns
    (table, scalars, n_steps, time_span, mean_abs_residual, var_decomp).
    Table rows are (key, mean, std, p10, p90, share_of_wall_pct_or_None);
    TPS is a rate, not a phase, so it gets no wall share.

    var_decomp answers "where does wall-clock variance come from": rows of
    (phase, 100*var(phase)/var(wall)) plus a final ("covariance remainder",
    pct) so the section sums to 100 — phase variances alone miss the
    cross-covariance (correlated phases) and untimed-gap contributions.
    None when var(wall) is ~0."""
    window = _window(rows, start_step, end_step)
    series = _collect_series(window)

    # Derived per-step series (not sum-of-aggregates: std and percentiles of
    # a sum are not the sum of stds/percentiles). Only when the run doesn't
    # already log the r0-era aggregate.
    if FORWARD_LOSS not in series and all(
        p in series for p in FORWARD_LOSS_PARTS
    ):
        parts = [series[p] for p in FORWARD_LOSS_PARTS]
        if len({len(p) for p in parts}) == 1:
            series[FORWARD_LOSS_DERIVED] = [sum(vals) for vals in zip(*parts)]

    wall_mean = statistics.fmean(series[WALL]) if WALL in series else None
    time_keys = sorted(k for k in series if k.startswith(TIME_PREFIX) and k != TPS)
    time_keys.sort(key=lambda k: k == WALL)  # wall_clock printed last
    table_keys = list(time_keys)
    if FORWARD_LOSS_DERIVED in series:
        table_keys.insert(max(len(table_keys) - 1, 0), FORWARD_LOSS_DERIVED)
    table = []
    for key in table_keys:
        vals = sorted(series[key])
        share = None
        if key != WALL and wall_mean:
            share = 100.0 * statistics.fmean(vals) / wall_mean
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        table.append(
            (key, statistics.fmean(vals), std,
             percentile(vals, 0.1), percentile(vals, 0.9), share)
        )

    var_decomp = None
    wall_vals = series.get(WALL, [])
    if len(wall_vals) > 1 and statistics.variance(wall_vals) > 1e-12:
        wall_var = statistics.variance(wall_vals)
        var_decomp = [
            (key, 100.0 * statistics.variance(series[key]) / wall_var)
            for key in time_keys
            if key != WALL and len(series[key]) > 1
        ]
        var_decomp.append(
            ("covariance remainder", 100.0 - sum(pct for _, pct in var_decomp))
        )

    scalars = {k: statistics.fmean(series[k]) for k in SCALARS if k in series}
    mean_abs_residual = (
        statistics.fmean([abs(v) for v in series[RESIDUAL]])
        if RESIDUAL in series
        else None
    )
    timestamps = [r["_timestamp"] for r in window if "_timestamp" in r]
    span = (min(timestamps), max(timestamps)) if timestamps else None
    return table, scalars, len(window), span, mean_abs_residual, var_decomp


def build_report(rows, start_step, end_step, last_n=20):
    """Pure builder for Sections 2-4. Returns (sections, meta).

    sections: ordered list of (title, [entry...]); entry is
    (label, window, stats, bundle) where bundle is a _bundle dict (full
    stats) or None ('n/a'). meta carries window sizes, the residual max-|.|
    for the harness caveat, and the steady/tail identity means.

    Reads ALL namespaces from history (no time/* filter); absent keys (older
    runs predating straggler/) yield None bundles, never crashes."""
    steady_rows = _window(rows, start_step, end_step)
    tail_rows = _last_n(rows, last_n)
    steady = _collect_series(steady_rows)
    tail = _collect_series(tail_rows)
    by_window = {"steady": steady, "last_20": tail}

    def entry(label, key, window, stats):
        return (label, window, stats, _bundle(by_window[window].get(key)))

    section2 = [entry(*spec) for spec in SECTION2_SPECS]

    # Section 3: residual reported as mean(signed) + max(|.|) so a large
    # NEGATIVE residual (phases exceeding wall = double-count) still trips
    # the caveat — a signed max() would miss it.
    resid = steady.get("check/timing_residual_frac")
    if resid:
        resid_bundle = {
            "mean": statistics.fmean(resid),
            "max": max(abs(v) for v in resid),
        }
    else:
        resid_bundle = None
    section3 = [
        ("check/timing_residual_frac (max=|.|)", "steady", ("mean", "max"),
         resid_bundle)
    ]
    section3 += [entry(*spec) for spec in SECTION3_IDENTITY_SPECS]

    # Section 4: reward/format get steady mean, last_20 mean, and the delta
    # (start-vs-end); loss is steady-only context.
    section4 = []
    for key in ("train/reward_mean", "train/format_rate"):
        b_steady = _bundle(steady.get(key))
        b_tail = _bundle(tail.get(key))
        section4.append((f"{key} [steady]", "steady", ("mean",), b_steady))
        section4.append((f"{key} [last_20]", "last_20", ("mean",), b_tail))
        delta = (
            None
            if b_steady is None or b_tail is None
            else {"mean": b_tail["mean"] - b_steady["mean"]}
        )
        section4.append(
            (f"{key} Δ(last_20 - steady)", "last_20-steady", ("mean",), delta)
        )
    section4.append(
        ("train/loss [steady]", "steady", ("mean",), _bundle(steady.get("train/loss")))
    )

    meta = {
        "n_steady": len(steady_rows),
        "n_tail": len(tail_rows),
        "residual_max_abs": resid_bundle["max"] if resid_bundle else None,
    }
    sections = [
        ("Section 2 — Do stragglers get worse? (steady_window)", section2),
        ("Section 3 — Are the numbers trustworthy? (the harness)", section3),
        ("Section 4 — Did the run learn? (steady_window vs last_20)", section4),
    ]
    return sections, meta


def gpu_metric(events, span, suffix):
    """Pool system-stream samples whose key matches system.gpu.<i>.<suffix>
    (e.g. 'gpu' = utilization, 'memory' = time accessing memory), restricted
    to the window's wall-time span. memoryAllocated etc. are excluded by the
    end-anchored match. Returns (bundle_or_None, sorted matched keys)."""
    pat = re.compile(rf"system\.gpu\.\d+\.{re.escape(suffix)}$")
    matched, vals = set(), []
    for e in events:
        ts = e.get("_timestamp", 0)
        if span is not None and not (span[0] <= ts <= span[1]):
            continue
        for key, val in e.items():
            if isinstance(val, (int, float)) and pat.match(key):
                matched.add(key)
                vals.append(val)
    return _bundle(vals), sorted(matched)


def print_section(title, entries):
    """Render a Section 2-4 entry list: requested stats shown, '-' for
    unrequested columns, 'n/a' across the row when the metric is absent."""
    print(f"\n=== {title} ===")
    print(f"{'metric':<40} " + " ".join(f"{c:>11}" for c in STAT_COLS))
    for label, _window, stats, bundle in entries:
        if bundle is None:
            cells = " ".join(f"{'n/a':>11}" for _ in STAT_COLS)
        else:
            cells = " ".join(
                f"{(_fmt(bundle[c]) if (c in stats and c in bundle) else '-'):>11}"
                for c in STAT_COLS
            )
        print(f"{label:<40} {cells}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="entity/project/run_id")
    parser.add_argument("--start-step", type=int, default=50)
    parser.add_argument("--end-step", type=int, default=100)
    parser.add_argument("--csv", action="store_true")
    args = parser.parse_args()

    import wandb

    run = wandb.Api().run(args.run)
    rows = list(run.scan_history())
    table, scalars, n, span, mean_abs_residual, var_decomp = aggregate_window(
        rows, args.start_step, args.end_step
    )
    if n == 0:
        sys.exit(f"no steps in [{args.start_step}, {args.end_step}) for {args.run}")
    sections, meta = build_report(rows, args.start_step, args.end_step)

    print(
        f"steady_window: steps [{args.start_step}, {args.end_step}) "
        f"-> {n} steps of {args.run}  (last_20: {meta['n_tail']} tail steps)"
    )
    print("\n=== Section 1 — Where does wall-clock time go? (steady_window) ===")
    print(
        f"{'metric':<46} {'mean_s':>9} {'std_s':>9} "
        f"{'p10_s':>9} {'p90_s':>9} {'% of wall':>10}"
    )
    for key, mean, std, p10, p90, share in table:
        share_str = f"{share:.1f}" if share is not None else "-"
        print(
            f"{key:<46} {mean:>9.3f} {std:>9.3f} "
            f"{p10:>9.3f} {p90:>9.3f} {share_str:>10}"
        )

    if var_decomp is not None:
        print("\nwall-clock variance decomposition (var(phase)/var(wall), %):")
        for key, pct in var_decomp:
            print(f"  {key:<32} {pct:>6.1f}")
    else:
        print("\nwall-clock variance ~0 in window — variance decomposition skipped")
    if TPS in scalars:
        print(f"{TPS}: {scalars[TPS]:.1f}")
    for key in ("train/reward_mean", "train/format_rate"):
        if key in scalars:
            print(f"{key}: {scalars[key]:.3f}")

    if mean_abs_residual is not None and mean_abs_residual > 0.05:
        print(
            f"\n!!! CAVEAT: mean |timing residual| = {mean_abs_residual:.4f} "
            "> 0.05 in this window. The timing harness was broken here "
            "(standing check #2) — the phase numbers above are NOT "
            "trustworthy and must not be cited.\n"
        )

    # Sections 2-4 (pure aggregation over all namespaces).
    for title, entries in sections:
        print_section(title, entries)
    if meta["residual_max_abs"] is not None and meta["residual_max_abs"] >= 0.05:
        print(
            f"\n!!! CAVEAT: worst |timing residual| = {meta['residual_max_abs']:.4f} "
            ">= 0.05 in steady_window. Section 1 phase numbers are NOT "
            "trustworthy for this window (standing check #2).\n"
        )

    # Section 5 — system stream, narrative only.
    print("\n=== Section 5 — How busy was the GPU? ===")
    print(
        "nvidia-smi system metrics (busy != useful, per CLAUDE.md) — "
        "narrative context only"
    )
    gpu_bundles = {}
    try:
        events = run.history(stream="events", samples=4000, pandas=False)
        for label, suffix in GPU_METRICS:
            bundle, matched = gpu_metric(events, span, suffix)
            gpu_bundles[label] = bundle
            if bundle:
                print(
                    f"  {label:<26} mean {_fmt(bundle['mean'])}  "
                    f"p10 {_fmt(bundle['p10'])}  p90 {_fmt(bundle['p90'])}  "
                    f"[keys: {', '.join(matched)}]"
                )
            else:
                print(f"  {label:<26} n/a (no matching keys in window)")
    except Exception as exc:  # absent stream must not kill the analysis
        print(f"  system-metrics stream unavailable ({exc}) — Section 5 n/a")

    if args.csv:
        out_dir = Path(__file__).resolve().parent.parent / "results" / "steady_state"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Human-readable artifact name: wandb run name first, id for dedup.
        run_id = args.run.split("/")[-1]
        out_path = out_dir / f"{run.name or 'run'}-{run_id}.csv"
        cols = ["mean", "std", "min", "max", "p10", "p90"]

        def row(metric, window, bundle, share=""):
            def g(s):
                return "" if (bundle is None or s not in bundle) else round(bundle[s], 6)
            return [metric, window] + [g(c) for c in cols] + [share]

        steady_label = f"[{args.start_step},{args.end_step})"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "window"] + cols + ["share_of_wall_pct"])
            # Section 1 (phase table) — min/max blank, share filled.
            for key, mean, std, p10, p90, share in table:
                b = {"mean": mean, "std": std, "p10": p10, "p90": p90}
                writer.writerow(
                    row(key, steady_label, b, "" if share is None else round(share, 3))
                )
            for key, pct in var_decomp or []:
                writer.writerow(row(f"var_share/{key}", steady_label, {"mean": pct}))
            for key, val in scalars.items():
                writer.writerow(row(key, steady_label, {"mean": val}))
            # Sections 2-4.
            for _title, entries in sections:
                for label, window, _stats, bundle in entries:
                    writer.writerow(row(label, window, bundle))
            # Section 5 (system stream).
            for label, bundle in gpu_bundles.items():
                writer.writerow(row(label, "events-stream", bundle))
            # Meta.
            writer.writerow(row("n_steady_steps", steady_label, {"mean": n}))
            writer.writerow(row("n_last20_steps", "last_20", {"mean": meta["n_tail"]}))
            if mean_abs_residual is not None:
                writer.writerow(
                    row("mean_abs_timing_residual", steady_label,
                        {"mean": mean_abs_residual})
                )
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
