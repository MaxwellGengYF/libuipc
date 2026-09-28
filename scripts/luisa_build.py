"""Build/run helper for the libuipc luisa backend (MSVC env + vcpkg toolchain).

Usage:
    python scripts/luisa_build.py build --target backend_luisa
    python scripts/luisa_build.py run
    python scripts/luisa_build.py ctest [args...]

Everything goes through `cmd /c` with vcvars64.bat sourced, because MSYS bash
mangles `cmd //c` arguments.
"""
import os
import subprocess
import sys

REPO = r"C:\dev\libuipc"
VCVARS = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
TEST_EXE = r"C:\dev\libuipc\build\Release\bin\uipc_test_backend_luisa.exe"


def script(body: str) -> int:
    bat = os.path.join(os.environ.get("TEMP", "."), "uipc_luisa_build.bat")
    with open(bat, "w") as f:
        f.write("@echo off\n")
        f.write("set VSLANG=1033\n")
        f.write("call \"%s\" >nul 2>&1\n" % VCVARS)
        f.write("cd /d %s\n" % REPO)
        f.write(body + "\n")
        f.write("exit /b %ERRORLEVEL%\n")
    return subprocess.call(["cmd", "/c", bat])


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    cmd = args[0]
    if cmd == "build":
        body = "cmake --build build --target %s -j 12" % " ".join(args[1:] or ["backend_luisa"])
    elif cmd == "run":
        body = "\"%s\" %s" % (TEST_EXE, " ".join(args[1:]))
    elif cmd == "ctest":
        body = "ctest --test-dir build -R backend_luisa --output-on-failure " + " ".join(args[1:])
    elif cmd == "sh":
        body = " ".join(args[1:])
    else:
        print("unknown command", cmd)
        return 1
    print("[luisa_build]", body)
    rc = script(body)
    print("[luisa_build] rc =", rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
