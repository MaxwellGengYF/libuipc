"""Compile-check helper for the libuipc luisa backend (MSVC + LC via vcpkg env).

Reuses the *exact* flags CMake produced for the `backend_luisa` test target
(parsed from build/compile_commands.json) so a syntax/instantiation check sees
the same include paths, defines and language settings as the real build.

Usage:
    python scripts/luisa_check_compile.py <source.cpp> [--syntax] [--filter SUBSTR]

--syntax   pass /Zs (parse + semantic analysis only, no object file)
--filter   which compile_commands entry to borrow the flags from
"""
import json
import os
import subprocess
import sys

REPO = r"C:\dev\libuipc"
VCVARS = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"


def borrowed_flags(filter_substr: str):
    db = json.load(open(os.path.join(REPO, "build", "compile_commands.json")))
    for e in db:
        if filter_substr in e["file"]:
            return e["command"], e["directory"]
    raise SystemExit(f"no compile_commands entry matching {filter_substr!r}")


def main():
    args = [a for a in sys.argv[1:]]
    syntax = "--syntax" in args
    if syntax:
        args.remove("--syntax")
    filt = "luisa_device_tests.cpp"
    if "--filter" in args:
        i = args.index("--filter")
        filt = args[i + 1]
        del args[i : i + 2]
    src = os.path.abspath(args[0])

    command, cwd = borrowed_flags(filt)
    # Strip the original compile line: keep flags only, drop /Fd /Fo and the input.
    toks = command.split()
    kept = []
    skip = False
    compiler = None
    for t in toks:
        if skip:
            skip = False
            continue
        if t in ("/Fd", "/Fo"):
            skip = True
            continue
        if t.startswith("/Fd") or t.startswith("/Fo"):
            continue
        if t.endswith(".cpp") or t.endswith(".cpp.obj"):
            continue
        if t == "/FS":
            continue
        if t.endswith("cl.exe"):
            compiler = t
            continue
        kept.append(t)

    if compiler is None:
        raise SystemExit("could not find cl.exe in the borrowed command line")
    kept.insert(0, compiler)
    if syntax:
        kept.insert(1, "/Zs")
    kept.append(src)

    bat = os.path.join(os.environ.get("TEMP", "."), "uipc_check_luisa.bat")
    with open(bat, "w") as f:
        f.write("@echo off\n")
        f.write("set VSLANG=1033\n")
        f.write(f'call "{VCVARS}" >nul 2>&1\n')
        f.write("cd /d " + cwd + "\n")
        f.write(" ".join('"%s"' % t if " " in t else t for t in kept) + "\n")
        f.write("exit /b %ERRORLEVEL%\n")

    print("[luisa_check_compile] compiling", src, "(syntax only)" if syntax else "")
    rc = subprocess.call(["cmd", "/c", bat])
    print("[luisa_check_compile] rc =", rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
