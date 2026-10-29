#!/usr/bin/env python3
"""Measure the OP-C3 CCD budget (optimize_plan.md OP-C3) on the filter test suite.

The claim under test is *not* "a bigger budget is faster" -- it is that the port can
afford the reference's convergence-sized budget at all now that the CCD loop is a
runtime ``$while`` instead of a trace-time unroll.  Three numbers per configuration:

* **cold wall** -- the first run after the configuration's cache entries were removed,
  i.e. a run that really pays the JIT for the trajectory-filter kernels;
* **new cache entries / PTX bytes** -- the size of the code that JIT produced, which is
  what used to blow up (the unrolled 64-iteration body took >900 s to JIT, see
  ``luisa_device/tuning.h``);
* **warm wall** -- the median of N runs with a warm cache, i.e. the *kernel cost* of the
  larger budget (a query that used to be truncated at 16 now iterates further).

Usage:
    python scripts/measure_ccd_bound.py [--tag bvh_filter] [--bounds 16,64] [--reps 3]

Writes ``benchmarks/results/<date>/ccd_bound.csv`` and prints a table.

NOTE ON COST.  This is the *whole-cache* variant: it freezes ``<exe>/.cache``, lets the
configuration build its own entries, and restores the frozen cache afterwards.  That is a
true cold measurement but it JITs every kernel the tag uses (~4 minutes per configuration on
an RTX 4060).  The numbers in
``agent_docs/performance/2026-10-28-op-c3-runtime-ccd-loop.md`` come from a cheaper
*surgical* variant: run two bounds once each on a warm cache, diff the cache entry names to
find the kernels the bound actually affects (exactly the four TOI kernels), delete only those
entries for a given bound, and subtract the warm run from the resulting run.  Both variants
measure the same thing; the surgical one has much less process-startup noise in the delta.
"""
from __future__ import annotations

import argparse
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXE = REPO / "build" / "windows" / "x64" / "release" / "uipc_test_backend_luisa.exe"
CACHE = EXE.parent / ".cache"
WORK = EXE.parent / ".cache_ccd_bound"


def run(tag: str, bound: int, timeout: int = 3600) -> tuple[float, str]:
    """One process: returns (wall seconds, stdout+stderr)."""
    env = dict(os.environ)
    env["UIPC_LUISA_CCD_MAX_ITER"] = str(bound)
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(EXE), f"[{tag}]"],
        env=env,
        cwd=str(EXE.parent),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    wall = time.perf_counter() - t0
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise SystemExit(f"[{tag}] bound={bound} failed (rc={proc.returncode}):\n{out[-4000:]}")
    return wall, out


def snapshot() -> dict[str, int]:
    return {p.name: p.stat().st_size for p in CACHE.glob("kernel_*.ptx")}


def take_new(before: dict[str, int], into: Path) -> dict[str, int]:
    """Move the entries this run created out of the cache; return them."""
    now = snapshot()
    fresh = {k: v for k, v in now.items() if k not in before}
    into.mkdir(parents=True, exist_ok=True)
    for name in fresh:
        shutil.move(str(CACHE / name), str(into / name))
        meta = CACHE / (name + ".metadata")
        if meta.exists():
            shutil.move(str(meta), str(into / (name + ".metadata")))
    return fresh


def restore(into: Path) -> None:
    if not into.is_dir():
        return
    for p in into.iterdir():
        shutil.move(str(p), str(CACHE / p.name))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bvh_filter")
    ap.add_argument("--bounds", default="16,64")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    if not EXE.exists():
        raise SystemExit(f"missing {EXE}; build the backend test target first")
    if not CACHE.is_dir():
        raise SystemExit(f"missing {CACHE}")

    bounds = [int(b) for b in args.bounds.split(",") if b.strip()]
    rows = []
    for bound in bounds:
        # A configuration's cache entries can only be identified by *recompiling*, so
        # the cold measurement freezes the whole cache (rename), lets this configuration
        # build its own, and then puts the frozen cache back.  Renaming is safe: nothing
        # in the directory is read by any other process and LC only looks it up at
        # runtime-init.
        frozen = CACHE.parent / ".cache.frozen"
        if frozen.exists():
            shutil.rmtree(frozen)
        CACHE.rename(frozen)
        CACHE.mkdir()
        try:
            print(f"== bound {bound}: cold run (empty cache)", flush=True)
            cold_wall, _ = run(args.tag, bound)
            fresh = snapshot()
        finally:
            this_one = WORK / str(bound)
            if this_one.exists():
                shutil.rmtree(this_one)
            shutil.copytree(CACHE, this_one)
            shutil.rmtree(CACHE)
            frozen.rename(CACHE)

        ptx_bytes = sum(v for k, v in fresh.items() if k.endswith(".ptx"))
        largest = max((v for k, v in fresh.items() if k.endswith(".ptx")), default=0)
        print(
            f"   cold {cold_wall:7.2f} s | {len(fresh) // 2} kernels | "
            f"{ptx_bytes / 1024.0:8.1f} KiB | largest {largest / 1024.0:7.1f} KiB",
            flush=True,
        )

        print(f"== bound {bound}: warm runs x{args.reps}", flush=True)
        warm = []
        for _ in range(args.reps):
            w, _ = run(args.tag, bound)
            warm.append(w)
            print(f"   warm {w:7.2f} s", flush=True)
        rows.append(
            dict(
                bound=bound,
                cold_s=cold_wall,
                kernels=len(fresh) // 2,
                ptx_kib=ptx_bytes / 1024.0,
                largest_kib=largest / 1024.0,
                warm_med_s=statistics.median(warm),
                warm_min_s=min(warm),
                warm_max_s=max(warm),
            )
        )
        # The measured configuration's entries stay in the cache: this is the one the
        # binary ships with, and a later configuration has different IR keys, so its own
        # cold run is unaffected by them.

    out_dir = REPO / "benchmarks" / "results" / time.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / "ccd_bound.csv"
    with csv.open("w", encoding="utf-8", newline="") as f:
        f.write("tag,bound,cold_s,kernels,ptx_kib,largest_kib,warm_med_s,warm_min_s,warm_max_s,reps\n")
        for r in rows:
            f.write(
                f"{args.tag},{r['bound']},{r['cold_s']:.3f},{r['kernels']},{r['ptx_kib']:.1f},"
                f"{r['largest_kib']:.1f},{r['warm_med_s']:.3f},{r['warm_min_s']:.3f},"
                f"{r['warm_max_s']:.3f},{args.reps}\n"
            )

    print()
    print("| ccd_max_iter | cold JIT s | kernels | PTX KiB | largest KiB | warm median s |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['bound']} | {r['cold_s']:.2f} | {r['kernels']} | {r['ptx_kib']:.1f} | "
            f"{r['largest_kib']:.1f} | {r['warm_med_s']:.3f} |"
        )
    print(f"\nwrote {csv.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
