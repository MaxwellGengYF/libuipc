#!/usr/bin/env python3
"""OP-C1 block-size sweep over the SpMV/BLAS kernel family (optimize_plan.md OP-C1).

Vehicle: the backend test suite's `spmv_large_timing` case (the K1 shape of
bench_plan.md §4.2: symmetric block BCOO, 1000 rows x 3 dofs, ~100k block entries) plus
its BLAS dot at n = 3000.  A dedicated Tier-1 binary would be the right home for this
(see `apps/benchmarks/tier1/t1_cases.cpp`), but the timing case already exists, already
loads the *product* kernel, and prints one line per variant -- so the sweep is honest
even though the harness is the test binary rather than the tier binary.

The block size is a *trace-time* value (`lcrt::set_block_size`), so each configuration is
a fresh process: that is the A/B protocol of optimize_plan.md §7 (same binary, one env
var, fresh process per run), not a shortcut.

Usage
-----
    python scripts/bench_blocksize_sweep.py                       # {64,128,192,256} x 3
    python scripts/bench_blocksize_sweep.py --blocks 64,128,256 --reps 5
    python scripts/bench_blocksize_sweep.py --case "[spmv]"       # include the correctness cases

Writes ``benchmarks/results/<date>/blocksize_spmv.csv`` and prints a median table.
Every configuration is also a *correctness* run: the case's own CHECKs (and, with
``--case "[spmv]"``, the whole SpMV/BLAS suite) must pass, which is what makes a retuned
block size trustworthy rather than merely fast.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXE = REPO / "build" / "windows" / "x64" / "release" / "uipc_test_backend_luisa.exe"

TIMING = re.compile(r"\[spmv timing\]\s+(\S+)\s+([0-9.]+)\s+ms/call")
PASSED = re.compile(r"All tests passed \((\d+) assertions in (\d+) test cases\)")


def run(block: int, case: str, timeout: int = 1800) -> tuple[dict[str, float], tuple[int, int] | None, str]:
    env = dict(os.environ)
    env["UIPC_LUISA_BLOCK_DIM"] = str(block)
    proc = subprocess.run([str(EXE), case], env=env, cwd=str(EXE.parent),
                          capture_output=True, text=True, timeout=timeout)
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise SystemExit(f"block {block} case {case} failed (rc={proc.returncode}):\n"
                         f"{out[-4000:]}")
    times = {m.group(1): float(m.group(2)) for m in TIMING.finditer(out)}
    if not times:
        raise SystemExit(f"block {block}: no '[spmv timing]' lines in the output:\n{out[-3000:]}")
    passed = PASSED.search(out)
    counts = (int(passed.group(1)), int(passed.group(2))) if passed else None
    return times, counts, out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blocks", default="64,128,192,256",
                    help="block sizes to sweep (multiples of 32)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--case", default="spmv_large_timing",
                    help='test case selector; "[spmv]" also runs the correctness suite')
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if not EXE.exists():
        raise SystemExit(f"missing {EXE}; build `backend_luisa` first")
    blocks = [int(b) for b in args.blocks.split(",") if b.strip()]
    out_dir = args.out or (REPO / "benchmarks" / "results" / _dt.date.today().isoformat())

    samples: dict[int, dict[str, list[float]]] = {}
    correctness: dict[int, tuple[int, int] | None] = {}
    for block in blocks:
        samples[block] = {}
        correctness[block] = None

    # Interleaved, not block-major: the clock drifts over a session, and a block-major
    # sweep would attribute that drift to whichever configuration ran last.  Every rep
    # touches every configuration once, which is the A/B/A/B shape of optimize_plan.md
    # section 7 applied to a parameter sweep.
    for r in range(args.reps):
        for block in blocks:
            times, counts, _ = run(block, args.case)
            for name, ms in times.items():
                samples[block].setdefault(name, []).append(ms)
            if counts:
                correctness[block] = counts
            print(f"rep {r + 1}/{args.reps} block {block:3d}: "
                  + "  ".join(f"{k}={v:.3f}" for k, v in times.items()), flush=True)

    kernels = sorted({k for per in samples.values() for k in per})
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "blocksize_spmv.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        f.write("kernel,block,median_ms,min_ms,max_ms,reps,assertions,test_cases\n")
        for kernel in kernels:
            for block in blocks:
                v = samples[block].get(kernel) or []
                if not v:
                    continue
                c = correctness[block]
                f.write(f"{kernel},{block},{statistics.median(v):.4f},{min(v):.4f},"
                        f"{max(v):.4f},{len(v)},{c[0] if c else ''},{c[1] if c else ''}\n")

    print()
    header = "| kernel | " + " | ".join(f"block {b}" for b in blocks) + " | best | vs 256 |"
    print(header)
    print("|---|" + "---:|" * (len(blocks) + 2))
    baseline = 256 if 256 in blocks else blocks[-1]
    for kernel in kernels:
        cells = []
        medians = {}
        for block in blocks:
            v = samples[block].get(kernel) or []
            if not v:
                cells.append("—")
                continue
            medians[block] = statistics.median(v)
            cells.append(f"{medians[block]:.3f}")
        best = min(medians, key=medians.get) if medians else None
        vs = (f"{medians[best] / medians[baseline]:.3f}" if best is not None
              and baseline in medians else "—")
        print(f"| {kernel} | " + " | ".join(cells) + f" | {best} | {vs} |")

    print()
    # Noise floor: `cpu_sym_spmv` is host code and `dot` at n = 3000 is traced with the
    # fixed reduction block (kBlock), so the knob cannot change either.  Their spread
    # across configurations *is* this vehicle's noise floor -- read every other row
    # against it before believing a difference.
    for control in ("cpu_sym_spmv", "dot"):
        v = [statistics.median(samples[b][control]) for b in blocks if samples[b].get(control)]
        if len(v) >= 2:
            print(f"noise floor ({control}): {min(v):.3f} .. {max(v):.3f} ms "
                  f"({(max(v) - min(v)) / min(v) * 100.0:.1f} % spread, code unaffected by the knob)")
    print()
    for block in blocks:
        c = correctness[block]
        print(f"block {block:3d}: {'pass' if c else 'no counter'} "
              + (f"({c[0]} assertions / {c[1]} cases)" if c else ""))
    print(f"\nwrote {csv_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
