#!/usr/bin/env python3
"""Aggregate uipc-bench/v1 records into the ratio table of bench_plan.md sections 8-9.

Reads every ``output/benchmark-runs/<case>-<backend>-<runid>.json`` produced by the
Tier-0/Tier-1 benchmark binaries, groups by (case, backend), and writes

    <out>/summary.csv      one row per (case, backend) + one ratio row per case
    <out>/summary.md       the same, readable, with the comparability checks of section 7.2

The aggregation is deliberately dumb: it medians the per-run medians and reports the
min/max spread. It never invents a number, and it prints every comparability violation
it can see (mixed backend_dll_sha256, mixed cache state, mixed lc_revision) because a
ratio computed across those is not a measurement.

Usage:
    python scripts/bench_aggregate.py [--dir output/benchmark-runs] [--out benchmarks/results/<date>]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load_records(directory: Path) -> list[dict]:
    records = []
    for path in sorted(directory.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:  # a truncated record must not pass silently
            print(f"[warn] {path.name}: invalid JSON ({exc}); skipped", file=sys.stderr)
            continue
        if data.get("schema") != "uipc-bench/v1":
            print(f"[warn] {path.name}: schema {data.get('schema')!r}; skipped", file=sys.stderr)
            continue
        records.append(data)
    return records


def comparability_notes(records: list[dict]) -> list[str]:
    """Section 7.2: two records are comparable only if these all agree."""
    notes = []

    def per_backend(field_path: tuple[str, ...]) -> dict[str, set]:
        out: dict[str, set] = defaultdict(set)
        for r in records:
            value = r
            for key in field_path:
                value = value.get(key, {}) if isinstance(value, dict) else {}
            out[r.get("backend", "?")].add(json.dumps(value, sort_keys=True))
        return out

    for backend, values in per_backend(("environment", "backend_dll_sha256")).items():
        if len(values) > 1:
            notes.append(f"{backend}: {len(values)} different backend_dll_sha256 values "
                         f"({', '.join(sorted(v.strip('\"')[:12] for v in values))}) -- "
                         f"records from different modules were mixed")
    for backend, values in per_backend(("environment", "lc_revision")).items():
        if len(values) > 1:
            notes.append(f"{backend}: mixed lc_revision {sorted(values)}")
    for backend, values in per_backend(("environment", "cache", "entries")).items():
        if len(values) > 1:
            notes.append(f"{backend}: cache entry count moved {sorted(values)} -- at least one "
                         f"run was cold (P6: never compare a cold and a warm run)")
    return notes


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(repo_root / "output" / "benchmark-runs"))
    ap.add_argument("--out", default=str(repo_root / "benchmarks" / "results" / "latest"))
    args = ap.parse_args()

    directory = Path(args.dir)
    out_dir = Path(args.out)
    records = load_records(directory)
    if not records:
        print(f"no uipc-bench/v1 records in {directory}", file=sys.stderr)
        return 1

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        grouped[(r["case"], r["backend"])].append(r)

    rows = []
    for (case, backend), runs in sorted(grouped.items()):
        medians = [r["median_ms"] for r in runs]
        per_unit = [r["per_unit_us"] for r in runs]
        rows.append(
            {
                "case": case,
                "backend": backend,
                "runs": len(runs),
                "tier": runs[0].get("tier", ""),
                "work_unit": runs[0].get("work_unit", ""),
                "work_units": runs[0].get("work_units", 0),
                "median_ms": statistics.median(medians),
                "median_min_ms": min(medians),
                "median_max_ms": max(medians),
                "per_unit_us": statistics.median(per_unit),
                "agreement_pct": statistics.median(
                    [r.get("agreement_pct", 0.0) for r in runs]),
                "checksum_ok": all(r.get("checksum_ok", False) for r in runs),
                "flags": "|".join(sorted({f for r in runs for f in r.get("flags", [])})),
                "dispatches_per_rep": statistics.median(
                    [r.get("counters", {}).get("dispatches_per_rep", 0.0) for r in runs]),
                "syncs_per_rep": statistics.median(
                    [r.get("counters", {}).get("syncs_per_rep", 0.0) for r in runs]),
                "graph_instantiate_ms": statistics.median(
                    [r.get("counters", {}).get("graph_instantiate_ms", 0.0) for r in runs]),
                "graph_update_ms": statistics.median(
                    [r.get("counters", {}).get("graph_update_ms", 0.0) for r in runs]),
            }
        )

    # Ratio table: luisa / cuda on the per-work-unit metric (section 7.4 decision rule).
    by_case: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_case[row["case"]][row["backend"]] = row

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["luisa_over_cuda"])
        writer.writeheader()
        for row in rows:
            ratio = ""
            pair = by_case[row["case"]]
            if "cuda" in pair and "luisa" in pair and pair["cuda"]["per_unit_us"] > 0:
                ratio = f"{pair['luisa']['per_unit_us'] / pair['cuda']['per_unit_us']:.3f}"
            writer.writerow({**row, "luisa_over_cuda": ratio})

    notes = comparability_notes(records)
    md = ["# Tier-0/1 benchmark summary", ""]
    md.append(f"- records: `{directory}` ({len(records)} files)")
    md.append(f"- generated from schema `uipc-bench/v1`; per-case JSON is the raw evidence")
    md.append("")
    if notes:
        md.append("## Comparability violations (section 7.2) -- do not quote the ratios below")
        md.append("")
        md.extend(f"- {n}" for n in notes)
        md.append("")
    md.append("## Ratio table (per work unit, median of per-run medians)")
    md.append("")
    md.append("| case | unit | cuda us/unit | luisa us/unit | luisa/cuda | verdict | agreement % | flags |")
    md.append("|---|---|---|---|---|---|---|---|")
    for case in sorted(by_case):
        pair = by_case[case]
        cuda = pair.get("cuda")
        luisa = pair.get("luisa")
        ratio = ""
        verdict = ""
        if cuda and luisa and cuda["per_unit_us"] > 0:
            r = luisa["per_unit_us"] / cuda["per_unit_us"]
            ratio = f"{r:.3f}"
            # section 7.4: parity 0.87..1.15, regressed > 1.15
            verdict = ("regressed" if r > 1.15 else
                       "faster" if r < 0.87 else "parity")
        agreement = f"{max((p['agreement_pct'] for p in pair.values()), default=0.0):.2f}"
        flags = "|".join(sorted({f for p in pair.values() for f in p["flags"].split("|") if f}))
        md.append(
            f"| {case} | {pair[next(iter(pair))]['work_unit']} | "
            f"{cuda['per_unit_us']:.4f} | {luisa['per_unit_us']:.4f} | {ratio} | {verdict} | "
            f"{agreement} | {flags} |"
        )
    md.append("")
    md.append("## Counter evidence (section 6.5)")
    md.append("")
    md.append("| case | backend | dispatches/rep | syncs/rep | graph instantiate ms | graph update ms |")
    md.append("|---|---|---|---|---|---|")
    for row in rows:
        md.append(
            f"| {row['case']} | {row['backend']} | {row['dispatches_per_rep']:.1f} | "
            f"{row['syncs_per_rep']:.1f} | {row['graph_instantiate_ms']:.3f} | "
            f"{row['graph_update_ms']:.3f} |"
        )
    md.append("")
    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    if notes:
        print(f"[warn] {len(notes)} comparability violation(s); see the summary header",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
