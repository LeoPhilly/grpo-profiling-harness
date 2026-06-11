#!/usr/bin/env python
"""Citable steady-state aggregates from a wandb run window.

Read-only analysis over logged history — touches no training code. The phase
table is only as trustworthy as standing check #2: a >5% mean |timing
residual| in the window prints a loud caveat.
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path

TIME_PREFIX = "time/"
WALL = "time/wall_clock"
TPS = "time/tokens_per_sec_generate"
RESIDUAL = "check/timing_residual_frac"
SCALARS = (TPS, "train/reward_mean", "train/format_rate")


def aggregate_window(rows, start_step, end_step):
    """Pure aggregation over history rows (dicts with _step). Returns
    (table, scalars, n_steps, time_span, mean_abs_residual) where table rows
    are (key, mean, std, share_of_wall_pct_or_None). TPS is a rate, not a
    phase, so it gets no wall share."""
    window = [
        r
        for r in rows
        if isinstance(r.get("_step"), (int, float))
        and start_step <= r["_step"] < end_step
    ]
    series = {}
    for row in window:
        for key, val in row.items():
            if isinstance(val, (int, float)):
                series.setdefault(key, []).append(val)

    wall_mean = statistics.fmean(series[WALL]) if WALL in series else None
    time_keys = sorted(k for k in series if k.startswith(TIME_PREFIX) and k != TPS)
    time_keys.sort(key=lambda k: k == WALL)  # wall_clock printed last
    table = []
    for key in time_keys:
        vals = series[key]
        share = None
        if key != WALL and wall_mean:
            share = 100.0 * statistics.fmean(vals) / wall_mean
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        table.append((key, statistics.fmean(vals), std, share))

    scalars = {k: statistics.fmean(series[k]) for k in SCALARS if k in series}
    mean_abs_residual = (
        statistics.fmean([abs(v) for v in series[RESIDUAL]])
        if RESIDUAL in series
        else None
    )
    timestamps = [r["_timestamp"] for r in window if "_timestamp" in r]
    span = (min(timestamps), max(timestamps)) if timestamps else None
    return table, scalars, len(window), span, mean_abs_residual


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
    table, scalars, n, span, mean_abs_residual = aggregate_window(
        rows, args.start_step, args.end_step
    )
    if n == 0:
        sys.exit(f"no steps in [{args.start_step}, {args.end_step}) for {args.run}")

    print(
        f"steady state: steps [{args.start_step}, {args.end_step}) "
        f"-> {n} steps of {args.run}"
    )
    print(f"{'metric':<34} {'mean_s':>9} {'std_s':>9} {'% of wall':>10}")
    for key, mean, std, share in table:
        share_str = f"{share:.1f}" if share is not None else "-"
        print(f"{key:<34} {mean:>9.3f} {std:>9.3f} {share_str:>10}")
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

    try:
        events = run.history(stream="events", samples=4000, pandas=False)
        key = "system.gpu.0.gpu"
        vals = sorted(
            e[key]
            for e in events
            if isinstance(e.get(key), (int, float))
            and (span is None or span[0] <= e.get("_timestamp", 0) <= span[1])
        )
        if vals:
            p10 = vals[int(0.1 * (len(vals) - 1))]
            p90 = vals[int(0.9 * (len(vals) - 1))]
            print(
                "nvidia-smi utilization (busy != useful, per CLAUDE.md) — "
                f"narrative context only: mean {statistics.fmean(vals):.1f}% "
                f"p10 {p10:.1f}% p90 {p90:.1f}%"
            )
        else:
            print("system-metrics stream: no GPU-util samples in window")
    except Exception as exc:  # absent stream must not kill the analysis
        print(f"system-metrics stream unavailable ({exc}) — GPU util skipped")

    if args.csv:
        out_dir = Path(__file__).resolve().parent.parent / "results" / "steady_state"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Human-readable artifact name: wandb run name first, id for dedup.
        run_id = args.run.split("/")[-1]
        out_path = out_dir / f"{run.name or 'run'}-{run_id}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "mean", "std", "share_of_wall_pct"])
            for key, mean, std, share in table:
                writer.writerow(
                    [key, round(mean, 6), round(std, 6),
                     "" if share is None else round(share, 3)]
                )
            for key, val in scalars.items():
                writer.writerow([key, round(val, 6), "", ""])
            writer.writerow(["n_steps", n, "", ""])
            writer.writerow(["window", f"[{args.start_step},{args.end_step})", "", ""])
            if mean_abs_residual is not None:
                writer.writerow(
                    ["mean_abs_timing_residual", round(mean_abs_residual, 6), "", ""]
                )
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
