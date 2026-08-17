@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -no_logo -arch=x64
if errorlevel 1 exit /b %errorlevel%

set "POC_ROOT=%~dp0.."
set "UPSTREAM=%POC_ROOT%\.work\Vulkan"
set "NINJA=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"

cmake -S "%UPSTREAM%" -B "%UPSTREAM%\build-gnn" -G Ninja -DCMAKE_MAKE_PROGRAM="%NINJA%" -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b %errorlevel%
cmake --build "%UPSTREAM%\build-gnn" --target gnncloth -j 8
if errorlevel 1 exit /b %errorlevel%

cmake -S "%POC_ROOT%\tests" -B "%POC_ROOT%\tests\build" -G Ninja -DCMAKE_MAKE_PROGRAM="%NINJA%" -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b %errorlevel%
cmake --build "%POC_ROOT%\tests\build" -j 8
if errorlevel 1 exit /b %errorlevel%
"%POC_ROOT%\tests\build\vgnn_format_test.exe" "%POC_ROOT%\model\artifacts\model.bin" "%POC_ROOT%\model\artifacts\golden.bin"
exit /b %errorlevel%
