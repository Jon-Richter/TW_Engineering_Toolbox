@echo off
setlocal

set "TOOLBOX_DIR=%~dp0.."
for %%A in ("%TOOLBOX_DIR%") do set "TOOLBOX_DIR=%%~fA"

set "VENDOR_DIR=%TOOLBOX_DIR%\vendor"
if not exist "%VENDOR_DIR%" mkdir "%VENDOR_DIR%"

set "PYTHON_EXE=%TOOLBOX_PYTHON_EXE%"
if "%PYTHON_EXE%"=="" set "PYTHON_EXE=python"
where "%PYTHON_EXE%" >nul 2>nul
if errorlevel 1 (
  echo Python not found. Set TOOLBOX_PYTHON_EXE or install Python 3.13.
  exit /b 2
)

set "PIP_DISABLE_PIP_VERSION_CHECK=1"
"%PYTHON_EXE%" -m pip install --upgrade pip
if errorlevel 1 exit /b %errorlevel%
"%PYTHON_EXE%" -m pip install -t "%VENDOR_DIR%" -r "%TOOLBOX_DIR%\requirements.bundle.txt"
endlocal & exit /b %errorlevel%
