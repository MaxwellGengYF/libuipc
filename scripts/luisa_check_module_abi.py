#!/usr/bin/env python3
"""Probe the backend module's *dynamic* ABI.

The in-tree tests link `uipc_backend_luisa.dll` directly (they import ~361 symbols), so
nothing in the test battery exercises the path a real consumer takes: `uipc::core`'s
`engine.cpp` `LoadLibrary`s the module and resolves the module's C entry points with
`get_function<...>("uipc_query_module")`.  This script closes that gap: it loads the DLL the
same way and reports which entry points are present, so an export-list regression cannot
pass unnoticed.

    python scripts/luisa_check_module_abi.py            # rc 0 iff every entry point resolves
    python scripts/luisa_check_module_abi.py --dll <path>

Entry points, in the order the loader tries them:
  * `uipc_query_module`  (src/core/core/internal/engine.cpp) -- required
  * `uipc_init_module`   (engine.cpp + sanity_checker.cpp)   -- required
  * `uipc_create_engine` / `uipc_destroy_engine` -- exported by the module as well, but the
    loader does not need them; they are reported informationally.
"""
from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DLL = ROOT / "build/Release/bin/uipc_backend_luisa.dll"
REQUIRED = ("uipc_query_module", "uipc_init_module")
OPTIONAL = ("uipc_create_engine", "uipc_destroy_engine")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dll", default=str(DEFAULT_DLL))
    args = ap.parse_args()
    path = Path(args.dll)
    if not path.exists():
        print(f"FAIL: {path} does not exist (build the backend first)")
        return 1

    dll = ctypes.WinDLL(str(path))  # the same LoadLibrary the core uses
    missing = []
    for name in REQUIRED:
        try:
            fn = getattr(dll, name)
            print(f"ok    {name:22s} (resolved, addr {ctypes.cast(fn, ctypes.c_void_p).value:#x})")
        except AttributeError:
            print(f"MISSING {name}")
            missing.append(name)
    for name in OPTIONAL:
        try:
            getattr(dll, name)
            print(f"info  {name:22s} (resolved)")
        except AttributeError:
            print(f"info  {name:22s} (absent -- the loader does not require it)")
    print("== module ABI:", "OK" if not missing else f"BROKEN (missing {', '.join(missing)}) ==")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
