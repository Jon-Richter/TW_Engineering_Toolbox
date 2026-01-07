@echo off
setlocal
REM Single entrypoint for the Engineering Toolbox (source or dist).
set "ROOT=%~dp0"
if not "%ROOT:~-1%"=="\" set "ROOT=%ROOT%\"

if "%LOCALAPPDATA%"=="" set "LOCALAPPDATA=%APPDATA%"
if "%LOCALAPPDATA%"=="" set "LOCALAPPDATA=%USERPROFILE%"
set "LOG_DIR=%LOCALAPPDATA%\EngineeringToolbox\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
for /f "usebackq" %%t in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"`) do set "TIMESTAMP=%%t"
set "LOG=%LOG_DIR%\launch_cmd_%TIMESTAMP%.log"
echo --- %DATE% %TIME% --- >> "%LOG%"
echo ROOT=%ROOT% >> "%LOG%"

set "PYTHON_EXE=%TOOLBOX_PYTHON_EXE%"
if "%PYTHON_EXE%"=="" set "PYTHON_EXE=python"
where "%PYTHON_EXE%" >nul 2>nul
if errorlevel 1 (
  echo Python not found. Set TOOLBOX_PYTHON_EXE or install Python 3.11+.
  echo Python not found. Set TOOLBOX_PYTHON_EXE or install Python 3.11+. >> "%LOG%"
  exit /b 2
)

set "DIST_DIR=%ROOT%dist"
set "DIST_READY="
if exist "%DIST_DIR%\EngineeringToolbox.pyz" if exist "%DIST_DIR%\run_extracted.py" set "DIST_READY=1"

set "VENDOR_DIR=%ROOT%vendor"
echo DIST_DIR=%DIST_DIR% >> "%LOG%"
echo VENDOR_DIR=%VENDOR_DIR% >> "%LOG%"
set "USE_VENDOR=%TOOLBOX_USE_VENDOR%"
if "%USE_VENDOR%"=="" set "USE_VENDOR=1"
set "VENDOR_READY="
if exist "%VENDOR_DIR%\*" set "VENDOR_READY=1"
if /I "%USE_VENDOR%"=="0" (
  set "VENDOR_READY="
  echo TOOLBOX_USE_VENDOR=0 (using local site-packages) >> "%LOG%"
)

if /I "%TOOLBOX_LAUNCH_MODE%"=="dist" goto launch_dist
if /I "%TOOLBOX_LAUNCH_MODE%"=="source" goto launch_source

if defined VENDOR_READY goto launch_source
if defined DIST_READY goto launch_dist

echo Missing vendor folder: %VENDOR_DIR%
echo Missing vendor folder: %VENDOR_DIR% >> "%LOG%"
echo Run scripts\vendorize_deps.bat on a build machine and include vendor\ in the distro.
echo Run scripts\vendorize_deps.bat on a build machine and include vendor\ in the distro. >> "%LOG%"
exit /b 2

:launch_source
echo Launching source mode. >> "%LOG%"
if defined VENDOR_READY (
  set "PYTHONPATH=%VENDOR_DIR%;%PYTHONPATH%"
  echo Using vendor: %VENDOR_DIR% >> "%LOG%"
) else (
  echo Using local site-packages only. >> "%LOG%"
)
echo PYTHON_EXE=%PYTHON_EXE% >> "%LOG%"
cd /d "%ROOT%"

REM Create a per-launch timestamped log to capture full stdout/stderr
for /f "usebackq delims=" %%t in (`powershell -NoProfile -Command "Get-Date -Format 'yyyyMMdd_HHmmss'"`) do set "TS=%%t"
set "LOG_TS=%LOG_DIR%\launch_cmd_%TS%.log"
echo --- %DATE% %TIME% --- >> "%LOG_TS%"
echo ROOT=%ROOT% >> "%LOG_TS%"
echo PYTHON_EXE=%PYTHON_EXE% >> "%LOG_TS%"
echo VENDOR_DIR=%VENDOR_DIR% >> "%LOG_TS%"
echo VENDOR_READY=%VENDOR_READY% >> "%LOG_TS%"
echo PYTHONPATH=%PYTHONPATH% >> "%LOG_TS%"

echo Running: %PYTHON_EXE% -u -m toolbox_app >> "%LOG_TS%"
"%PYTHON_EXE%" -u -m toolbox_app >> "%LOG_TS%" 2>&1
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
  echo Launch failed with code %EXITCODE%. >> "%LOG%"
  echo Launch failed with code %EXITCODE%. >> "%LOG_TS%"
  echo Launch failed. See log: %LOG_TS%
  REM Open a persistent console showing the log to help debugging
  start "" cmd /k "type "%LOG_TS%" & echo. & echo Press any key to close... & pause"
)
endlocal & exit /b %EXITCODE%

:launch_dist
if not defined DIST_READY (
  echo dist launcher not found under %DIST_DIR% >> "%LOG%"
  echo dist launcher not found under %DIST_DIR%
  endlocal & exit /b 2
)
echo Launching dist mode. >> "%LOG%"
"%PYTHON_EXE%" "%DIST_DIR%\run_extracted.py" >> "%LOG%" 2>&1
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
  echo Dist launch failed with code %EXITCODE%. >> "%LOG%"
  echo Dist launch failed. See log: %LOG%
)
endlocal & exit /b %EXITCODE%
