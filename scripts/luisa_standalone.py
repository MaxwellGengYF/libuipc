"""Configure + build + run the standalone luisa_device harness (MSVC + LC).

The harness links only LuisaCompute and the luisa_device sources -- no vcpkg
libraries -- so it is the lightest possible end-to-end check of the device layer
and the math shim.

Usage: python scripts/luisa_standalone.py [backend]
"""
import os
import subprocess
import sys

REPO = r"C:\dev\libuipc"
SRC = os.path.join(REPO, "apps", "tests", "backends", "luisa", "standalone")
BUILD = os.path.join(SRC, "build-msvc")
VCVARS = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    body = "\n".join(
        [
            'cmake -S "%s" -B "%s" -G Ninja -DCMAKE_BUILD_TYPE=Release ^' % (SRC, BUILD),
            "  -DLUISA_COMPUTE_ROOT=C:/dev/LuisaCompute ^",
            "  -DLUISA_COMPUTE_PREFIX=C:/dev/LuisaCompute/install-msvc ^",
            "  -DUIPC_VCPKG_INSTALLED_DIR=C:/dev/libuipc/build/vcpkg_installed",
            'if errorlevel 1 exit /b %errorlevel%',
            'cmake --build "%s" -j 12' % BUILD,
            'if errorlevel 1 exit /b %errorlevel%',
            'cd /d "%s"' % BUILD,
            'luisa_device_tests.exe %s' % backend,
        ]
    )
    bat = os.path.join(os.environ.get("TEMP", "."), "uipc_standalone.bat")
    with open(bat, "w") as f:
        f.write("@echo off\n")
        f.write("set VSLANG=1033\n")
        f.write('call "%s" >nul 2>&1\n' % VCVARS)
        f.write(body + "\n")
        f.write("exit /b %ERRORLEVEL%\n")
    rc = subprocess.call(["cmd", "/c", bat])
    print("[luisa_standalone] rc =", rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
