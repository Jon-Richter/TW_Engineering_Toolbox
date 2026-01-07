@echo off
setlocal
set "ROOT=%~dp0"
if not "%ROOT:~-1%"=="\" set "ROOT=%ROOT%\"
set "REQ=%ROOT%requirements.txt"

if not exist "%REQ%" (
  echo requirements.txt not found at %REQ%
  exit /b 2
)

set "PYTHON_EXE=%TOOLBOX_PYTHON_EXE%"
if "%PYTHON_EXE%"=="" set "PYTHON_EXE=py"
where "%PYTHON_EXE%" >nul 2>nul
if errorlevel 1 (
  set "PYTHON_EXE=python"
  where "%PYTHON_EXE%" >nul 2>nul
  if errorlevel 1 (
    echo Python not found. Set TOOLBOX_PYTHON_EXE or install Python 3.11+.
    exit /b 2
  )
)

"%PYTHON_EXE%" -m pip install -r "%REQ%"
endlocal & exit /b %errorlevel%
