@echo off
REM Rebuild the local LuisaCompute (incremental, ninja) and refresh the install
REM prefix that libuipc's luisa module links against.  Usage:
REM   scripts\lc_rebuild.cmd            build + install + copy DLLs next to our exes
REM   scripts\lc_rebuild.cmd build      build only
setlocal
set LC_ROOT=C:\dev\LuisaCompute
set LC_PREFIX=%LC_ROOT%\install-msvc
set UIPC_BIN=C:\dev\libuipc\build\windows\x64\release

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (echo [lc] vcvars64 failed & exit /b 1)

cd /d %LC_ROOT%
echo [lc] building %LC_ROOT%\build-msvc ...
cmake --build build-msvc --config Release
if errorlevel 1 (echo [lc] BUILD FAILED & exit /b 1)

if /I "%1"=="build" (echo [lc] build only & exit /b 0)

echo [lc] installing to %LC_PREFIX% ...
cmake --install build-msvc --prefix "%LC_PREFIX%"
if errorlevel 1 (echo [lc] INSTALL FAILED & exit /b 1)

echo [lc] refreshing DLLs next to libuipc binaries ...
copy /y "%LC_PREFIX%\bin\*.dll" "%UIPC_BIN%\" >nul
if errorlevel 1 (echo [lc] COPY FAILED & exit /b 1)
echo [lc] done
