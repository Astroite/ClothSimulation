@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -no_logo -arch=x64
if errorlevel 1 exit /b %errorlevel%
set "POC_ROOT=%~dp0.."
set "UPSTREAM=%POC_ROOT%\.work\Vulkan"
rem Refresh generated sources and validated shaders before invoking Ninja.
rem Otherwise edits in include/ can accidentally build a stale copied overlay.
pwsh -NoProfile -ExecutionPolicy Bypass -File "%POC_ROOT%\bootstrap.ps1" -SkipFetch
if errorlevel 1 exit /b %errorlevel%
set "NINJA=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
cmake -S "%UPSTREAM%" -B "%UPSTREAM%\build-mlclothcpu" -G Ninja -DCMAKE_MAKE_PROGRAM="%NINJA%" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PROJECT_INCLUDE="%POC_ROOT%/tools/msvc_dependencies.cmake"
if errorlevel 1 exit /b %errorlevel%
cmake --build "%UPSTREAM%\build-mlclothcpu" --target mlclothcpu -j 8
if errorlevel 1 exit /b %errorlevel%
if /I "%~1"=="--app-only" exit /b 0
cmake -S "%POC_ROOT%\tests" -B "%POC_ROOT%\tests\build" -G Ninja -DCMAKE_MAKE_PROGRAM="%NINJA%" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PROJECT_INCLUDE="%POC_ROOT%/tools/msvc_dependencies.cmake"
if errorlevel 1 exit /b %errorlevel%
cmake --build "%POC_ROOT%\tests\build" -j 8
if errorlevel 1 exit /b %errorlevel%
ctest --test-dir "%POC_ROOT%\tests\build" --output-on-failure
exit /b %errorlevel%

