#!/usr/bin/env python3
"""Patch LuisaCompute's CUDA device-math builtins so that the *double* overloads
use double-precision device functions instead of their single-precision `...f`
counterparts.

Why
---
`src/backends/cuda/cuda_builtin/cuda_device_math.h` maps every `lc_double`
transcendental to the float function, e.g.

    [[nodiscard]] __device__ inline lc_double lc_log(lc_double x) noexcept { return logf(x); }
    [[nodiscard]] __device__ inline lc_double lc_pow(lc_double x, lc_double a) noexcept { return powf_impl(x, a); }

and `powf_impl(lc_double, lc_double)` itself falls back to `powf(x, y)` for a
non-integral exponent.  The CUDA backend therefore evaluates double-precision
log/exp/pow/trig in **float** precision, i.e. roughly 1e-7 relative accuracy
instead of 1e-16, while the reference (nvcc-compiled) code uses the accurate
double functions.

The libuipc port was written against the reference's numerics, so this shows up as
a systematic device-vs-host deviation in the FEM/ABD constitutive code (measured:
1.3e-07 to 2.5e-06 relative per element for the rod/spring constitutions of
T5.2 batch c, where the host instantiation of the *same* routine agrees with the
oracle to ~1e-13).

The builtins are embedded into the runtime by `luisa_embed_device_lib` from the
`cuda_builtin/*.h` glob, so the patch takes effect after rebuilding LC (the
`cuda_builtin_embedded.cpp` byproduct is regenerated).  This is the second local
LC fix (the first is the stale-prebuilt export in cookbook 1.2).

Usage
-----
    python scripts/patch_lc_double_transcendentals.py [--lc-root C:/dev/LuisaCompute] [--check]

`--check` reports whether the patch is applied without writing anything.
Idempotent: running it twice is a no-op.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

REL_PATH = pathlib.Path("src/backends/cuda/cuda_builtin/cuda_device_math.h")

# `lc_double f(lc_double ...) { return <floatfn>(...); }` -> the double counterpart.
# Only the *double* one-liners are rewritten; the lc_float overloads keep their
# float functions (that is what they are for).
DOUBLE_ONELINER = re.compile(
    r"(?P<head>__device__\s+inline\s+lc_double\s+lc_(?P<name>\w+)\s*"
    r"\((?P<args>[^)]*)\)\s*noexcept\s*\{\s*return\s+)(?P<fn>\w+)f\((?P<callargs>[^)]*)\)(?P<tail>\s*;\s*\})"
)

# `powf_impl(lc_double ...)` must not fall back to the float `powf`.
POW_DOUBLE = re.compile(
    r"(?P<head>__device__\s+inline\s+lc_double\s+powf_impl\s*\([^)]*\)\s*noexcept\s*\{)"
    r"(?P<body>[^}]*?)(?P<tail>\})"
)


def patch_text(text: str) -> tuple[str, list[str]]:
    changes: list[str] = []

    def sub_oneliner(m: re.Match[str]) -> str:
        # `(?P<fn>\w+)f\(` consumes the trailing `f`, so a double-precision
        # counterpart is exactly the case `fn == name` (e.g. lc_acos -> acosf).
        # `lc_sqrt`/`lc_exp2`/`lc_fma` ... have no trailing `f` and never match;
        # `lc_abs -> fabs` is exact and is left alone.
        fn, name = m.group("fn"), m.group("name")
        if fn == name:
            changes.append(f"lc_{name}: {fn}f -> {name}")
            return f"{m.group('head')}{name}({m.group('callargs')}){m.group('tail')}"
        return m.group(0)

    text = DOUBLE_ONELINER.sub(sub_oneliner, text)

    def sub_pow(m: re.Match[str]) -> str:
        body = m.group("body")
        if "powf(x, y)" in body:
            changes.append("powf_impl(lc_double): powf -> pow")
            body = body.replace("powf(x, y)", "pow(x, y)")
        return f"{m.group('head')}{body}{m.group('tail')}"

    text = POW_DOUBLE.sub(sub_pow, text)
    return text, changes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lc-root", default="C:/dev/LuisaCompute")
    ap.add_argument("--check", action="store_true", help="report only, do not write")
    args = ap.parse_args()

    path = pathlib.Path(args.lc_root) / REL_PATH
    if not path.is_file():
        print(f"error: {path} not found", file=sys.stderr)
        return 2

    original = path.read_text(encoding="utf-8")
    patched, changes = patch_text(original)

    if args.check:
        if changes:
            print(f"NOT patched ({len(changes)} rewrite(s) pending):")
            for c in changes:
                print(f"  {c}")
            return 1
        print("already patched (no double-precision float fallbacks found)")
        return 0

    if not changes:
        print("already patched; nothing to do")
        return 0

    path.write_text(patched, encoding="utf-8", newline="")
    print(f"patched {path} ({len(changes)} rewrite(s)):")
    for c in changes:
        print(f"  {c}")
    print("\nrebuild LuisaCompute and reinstall it (the builtin embed regenerates "
          "from cuda_builtin/*.h), then rebuild libuipc.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
