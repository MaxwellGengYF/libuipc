#!/usr/bin/env python3
"""Generate `src/backends/luisa/luisa_module.def` -- the explicit export list of the
LuisaCompute backend module (`uipc_backend_luisa.dll`).

Why this file exists
--------------------
`WINDOWS_EXPORT_ALL_SYMBOLS` exports *every* symbol the module's objects define.  Because
each kernel signature instantiates a large amount of LuisaCompute DSL machinery, the
auto-generated `.def` reached **69295** entries in T6.3b3, and MSVC's export table is
capped at 65535 (`LINK : fatal error LNK1189: cannot have more than 64K exports`).  The
module therefore exports an explicit list: its C API plus every symbol the in-tree
consumers actually import.  Adding a test that uses a new backend symbol makes the *link*
fail with `unresolved external symbol` -- that is the intended, loud signal; re-run this
script and commit the refreshed file.

Recipe
------
    python scripts/luisa_module_def.py            # rewrite the .def
    python scripts/luisa_module_def.py --check     # rc 1 if the committed file is stale

The consumer set is derived mechanically: every PE file under the build output that has an
import table entry for `uipc_backend_luisa.dll` contributes its imported names.  The
module's own C entry points are always included (they are resolved with GetProcAddress at
runtime, so nothing imports them).
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_DLL = "uipc_backend_luisa.dll"
TARGET = ROOT / "src/backends/luisa/luisa_module.def"

# Resolved through the module's C ABI (GetProcAddress), never imported.
C_API = (
    "uipc_create_engine",
    "uipc_destroy_engine",
    "uipc_init_module",
    "uipc_query_module",
)

BANNER = """\
; ---------------------------------------------------------------------------
; {name} -- the explicit export list of {dll}
;
; GENERATED FILE -- do not edit by hand.  Regenerate with
;     python scripts/luisa_module_def.py
; (which also explains why the module cannot use WINDOWS_EXPORT_ALL_SYMBOLS: the
; auto-generated list exceeded MSVC's 65535-export ceiling -> LNK1189).
;
; Contents: the module's C API + every symbol the in-tree consumers import from it.
; {count} entries, generated from {consumers} consumer(s).
; ---------------------------------------------------------------------------
EXPORTS
"""


def find_dumpbin() -> str:
    env = os.environ.get("DUMPBIN")
    if env and Path(env).exists():
        return env
    for vs in (
        Path(r"C:\Program Files\Microsoft Visual Studio\2022"),
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022"),
    ):
        hits = sorted(vs.glob("*/VC/Tools/MSVC/*/bin/Hostx64/x64/dumpbin.exe"))
        if hits:
            return str(hits[-1])
    raise SystemExit("dumpbin.exe not found; pass --dumpbin or set DUMPBIN")


def imports_from_module(dumpbin: str, path: Path) -> set[str]:
    """The names `path` imports from `MODULE_DLL` (empty if it imports nothing from it).

    The import table is a sequence of sections, each opened by a line naming the DLL
    (e.g. `    uipc_backend_luisa.dll`) followed by `ordinal hint RVA name` lines; parsing
    therefore has to stop at the next section rather than scan to the end of the dump (the
    module's own name also occurs in the output header, which is why a naive scan reports a
    DLL as its own consumer).
    """
    out = subprocess.run([dumpbin, "/nologo", "/imports", str(path)],
                         capture_output=True, text=True, errors="replace")
    text = out.stdout + out.stderr
    section = re.compile(r"^\s+([A-Za-z0-9_.\-]+\.dll)\s*$")
    entry = re.compile(r"^\s*[0-9A-Fa-f]+\s+([?A-Za-z_][^\s]*)\s*$")
    found: set[str] = set()
    current: str | None = None
    for line in text.splitlines():
        m = section.match(line)
        if m:
            current = m.group(1).lower()
            continue
        if current == MODULE_DLL.lower():
            e = entry.match(line)
            if e:
                found.add(e.group(1))
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify the committed file (rc 1 if stale)")
    ap.add_argument("--build-dir", default=str(ROOT / "build"))
    ap.add_argument("--dumpbin", default=None)
    args = ap.parse_args()

    dumpbin = args.dumpbin or find_dumpbin()

    candidates: list[Path] = []
    for pattern in ("**/bin/*.exe", "**/bin/*.dll", "**/*.exe", "**/*.dll"):
        candidates += [Path(p) for p in glob.glob(os.path.join(args.build_dir, pattern),
                                                 recursive=True)]
    candidates = sorted({p for p in candidates if p.suffix.lower() in (".exe", ".dll")})

    symbols: set[str] = set(C_API)
    consumers: list[str] = []
    for path in candidates:
        try:
            got = imports_from_module(dumpbin, path)
        except Exception:  # a non-PE or a locked file: skip
            continue
        if got:
            consumers.append(path.name)
            symbols |= got

    body = BANNER.format(name=TARGET.name, dll=MODULE_DLL, count=len(symbols) + 1,
                         consumers=len(consumers))
    body += "".join(f"    {s}\n" for s in sorted(symbols))

    if args.check:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        same = current == body
        print(f"{'ok  ' if same else 'STALE'} {TARGET.relative_to(ROOT)} "
              f"({len(symbols)} symbols from {len(consumers)} consumer(s))")
        if not same:
            print("re-run: python scripts/luisa_module_def.py")
        return 0 if same else 1

    TARGET.write_text(body, encoding="utf-8")
    print(f"wrote {TARGET.relative_to(ROOT)}: {len(symbols)} symbols "
          f"(C API {len(C_API)}) from {len(consumers)} consumer(s):")
    for c in consumers:
        print(f"    {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
