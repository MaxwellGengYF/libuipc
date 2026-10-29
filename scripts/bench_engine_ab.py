#!/usr/bin/env python3
"""Engine A/B for the LuisaCompute frame command-list batching (optimize_plan.md OP-A1).

One binary, one scene, two configurations -- `config["luisa"]["frame_batching"]` is the
only switch, so the arms cannot differ in anything but the batching layer.  Each run is a
**fresh process** (bench_plan.md B3: a second Engine in one process breaks the LC stream),
and the runs are interleaved A/B/A/B/A/B so that a thermal or clock drift of the session
cannot land on one arm only.

Per run the driver reports its per-frame wall times (`round_ms`) and the engine's own
`last_frame_stats`, so every row carries both the clock and the *counters* (the honest
part: `commands` absorbed, `commits` really submitted, `direct_submits` that escaped the
frame list, `syncs`, and `compiles` -- OP-A4).

The physics gate is the state checksum: if the two arms' checksums differ the script
exits 1, because a batching change that moves a vertex is a bug, not an optimisation.

Usage
-----
    python scripts/bench_engine_ab.py                     # 3 rounds x 30 frames
    python scripts/bench_engine_ab.py --rounds 6 --frames 100 --warmup 10
    python scripts/bench_engine_ab.py --bin build/windows/x64/release/uipc_bench_engine_ab.exe

Writes ``benchmarks/results/<date>/engine_ab.csv`` and ``engine_ab.md``.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BIN = REPO / "build" / "windows" / "x64" / "release" / "uipc_bench_engine_ab.exe"


def run_once(binary: Path, batching: int, frames: int, warmup: int) -> dict:
    """One measured process.  Returns the driver's JSON plus the wall clock of the run."""
    cmd = [
        str(binary),
        f"--batching={batching}",
        f"--frames={frames}",
        f"--warmup={warmup}",
        "--json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(binary.parent))
    if proc.returncode != 0:
        raise SystemExit(
            f"run failed (rc={proc.returncode}): {' '.join(cmd)}\n"
            f"--- stdout ---\n{proc.stdout[-4000:]}\n--- stderr ---\n{proc.stderr[-4000:]}"
        )
    # The driver prints exactly one JSON object; anything before it is log noise.
    text = proc.stdout.strip()
    start = text.find("{")
    if start < 0:
        raise SystemExit(f"no JSON in the driver output:\n{text[-4000:]}")
    return json.loads(text[start:])


def counters_of(result: dict) -> dict:
    return (result.get("last_frame_stats") or {}).get("counters") or {}


def frame_medians(result: dict) -> list[float]:
    """Per-frame wall times of one run (``round_ms`` in the driver's schema)."""
    ms = result.get("round_ms") or []
    if not ms:
        raise SystemExit("driver reported no per-frame timings (round_ms missing)")
    return ms


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    ap.add_argument("--rounds", type=int, default=3, help="interleaved A/B rounds")
    ap.add_argument("--frames", type=int, default=30, help="measured frames per run")
    ap.add_argument("--warmup", type=int, default=3, help="warm-up frames per run (JIT, cache)")
    ap.add_argument("--out", type=Path, default=None, help="artifact directory")
    args = ap.parse_args(argv)

    if not args.bin.exists():
        raise SystemExit(f"missing {args.bin}; build the engine_ab target first")
    out_dir = args.out or (REPO / "benchmarks" / "results" / _dt.date.today().isoformat())

    summary: dict[str, dict] = {}
    results: dict[str, list[dict]] = {"on": [], "off": []}

    for r in range(args.rounds):
        for label, batching in (("on", 1), ("off", 0)):
            res = run_once(args.bin, batching, args.frames, args.warmup)
            results[label].append(res)
            med = statistics.median(frame_medians(res))
            c = counters_of(res)
            print(
                f"[{r + 1}/{args.rounds}] batching={batching:<2} median={med:.4f} ms  "
                f"commands={c.get('commands', 0)} commits={c.get('commits', 0)} "
                f"direct={c.get('direct_submits', 0)} compiles={c.get('compiles', 0)} "
                f"checksum={res.get('checksum')}",
                flush=True,
            )

    for label, rs in results.items():
        medians = [statistics.median(frame_medians(r)) for r in rs]
        counters = [counters_of(r) for r in rs]
        summary[label] = {
            "runs": len(rs),
            "median_ms": statistics.median(medians),
            "spread_pct": (
                (max(medians) - min(medians)) / statistics.median(medians) * 100.0
                if statistics.median(medians)
                else 0.0
            ),
            "checksums": [r.get("checksum") for r in rs],
            "commits": [c.get("commits", 0) for c in counters],
            "commands": [c.get("commands", 0) for c in counters],
            "direct_submits": [c.get("direct_submits", 0) for c in counters],
            "syncs": [c.get("syncs", 0) for c in counters],
            # OP-A4: compile *calls* per frame; must be 0 after the first frame.
            "compiles": [c.get("compiles", 0) for c in counters],
        }

    on, off = summary["on"], summary["off"]
    ratio = on["median_ms"] / off["median_ms"] if off["median_ms"] else float("nan")
    checksums = sorted({str(c) for c in on["checksums"] + off["checksums"]})
    checksum_ok = len(checksums) == 1

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "engine_ab.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "config",
                "runs",
                "median_ms",
                "spread_pct",
                "checksum",
                "commits_per_frame",
                "commands_per_frame",
                "direct_submits_per_frame",
                "syncs_per_frame",
                "compiles_per_frame",
            ]
        )
        for label, batching in (("on", 1), ("off", 0)):
            s = summary[label]
            w.writerow(
                [
                    f"frame_batching={batching}",
                    s["runs"],
                    f"{s['median_ms']:.6f}",
                    f"{s['spread_pct']:.2f}",
                    ";".join(sorted({str(c) for c in s["checksums"]})),
                    statistics.median(s["commits"]),
                    statistics.median(s["commands"]),
                    statistics.median(s["direct_submits"]),
                    statistics.median(s["syncs"]),
                    statistics.median(s["compiles"]),
                ]
            )

    md = [
        "# Engine A/B: frame command-list batching (optimize_plan.md OP-A1)",
        "",
        f"Binary: `{args.bin}`  ",
        f"{args.rounds} interleaved rounds x {args.frames} timed frames "
        f"({args.warmup} warm-up), one fresh process per run.",
        "",
        "| configuration | median ms/frame | spread over runs | commits/frame | "
        "commands/frame | direct submits/frame | syncs/frame | compiles/frame |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, batching in (("on", 1), ("off", 0)):
        s = summary[label]
        md.append(
            f"| `frame_batching={batching}` | {s['median_ms']:.4f} | {s['spread_pct']:.1f} % | "
            f"{statistics.median(s['commits'])} | {statistics.median(s['commands'])} | "
            f"{statistics.median(s['direct_submits'])} | {statistics.median(s['syncs'])} | "
            f"{statistics.median(s['compiles'])} |"
        )
    md += [
        "",
        f"Ratio (on/off) of the median frame: **{ratio:.3f}** "
        "(< 1 means batching is faster).",
        f"State checksum identical in both configurations: "
        f"**{'yes' if checksum_ok else 'NO -- the two arms diverged'}**"
        + (f" (`{checksums[0]}`)" if checksum_ok else f" ({', '.join(checksums)})"),
        "",
        "## How to read this",
        "",
        "* `commands/frame` -- kernels absorbed into the frame command list.",
        "* `commits/frame` -- command lists really submitted "
        "(the OP-A1 acceptance: O(1), not O(#kernels)).",
        "* `direct submits/frame` -- commands that escaped the list.  Exactly one is "
        "expected: the frame's host-sourced upload, which cannot be deferred "
        "(OP-A2's `record_upload`).",
        "* `compiles/frame` -- shader creations during the last frame.  After the warm-up "
        "frames this must be 0 (OP-A4: a lost `static` would show up here).",
        "* The wall clock at a ~0.2 ms frame is *not* separable from the run-to-run spread "
        "on this shell: report the counters as the measurement and treat the ratio as "
        "context, not as a claim (optimize_plan.md §7.1 requires a separating CI).",
    ]
    md_path = out_dir / "engine_ab.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print()
    print("\n".join(md[4:9]))
    print(f"\nwrote {csv_path.relative_to(REPO)} and {md_path.relative_to(REPO)}")
    return 0 if checksum_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
