@echo off
REM Helper: configure+build the T0.1 LC spike with MSVC + Ninja.
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 ( echo VCVARS_FAILED & exit /b 1 )
cd /d C:\dev\libuipc
cmake -S scratch\lc_spike -B scratch\lc_spike\build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLUISA_COMPUTE_ROOT=C:/dev/LuisaCompute
if errorlevel 1 ( echo CONFIGURE_FAILED & exit /b 1 )
cmake --build scratch\lc_spike\build
if errorlevel 1 ( echo BUILD_FAILED & exit /b 1 )
echo BUILD_OK
endlocal
