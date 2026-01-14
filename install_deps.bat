@echo off
setlocal
set "ROOT=%~dp0"
if not "%ROOT:~-1%"=="\" set "ROOT=%ROOT%\"
set "REQ=%ROOT%requirements.txt"

if not exist "%REQ%" (
  echo requirements.txt not found at %REQ%
  exit /b 2
)

set "PYTHON_EXE="
call :detect_python "%TOOLBOX_PYTHON_EXE%"
if not defined PYTHON_EXE call :detect_python py
if not defined PYTHON_EXE call :detect_python python
if not defined PYTHON_EXE call :detect_python python3
if not defined PYTHON_EXE (
  call :find_python_from_registry
)
if not defined PYTHON_EXE (
  echo Python not found. Set TOOLBOX_PYTHON_EXE, add Python to PATH, or install Python 3.11+.
  exit /b 2
)

"%PYTHON_EXE%" -m pip install --upgrade --user -r "%REQ%"
endlocal & exit /b %errorlevel%

:detect_python
setlocal
set "TRY=%~1"
if "%TRY%"=="" endlocal & exit /b 1
for /f "delims=" %%P in ('where "%TRY%" 2^>nul') do (
  endlocal
  set "PYTHON_EXE=%%~P"
  goto :eof
)
endlocal
exit /b 1

:find_python_from_registry
for %%H in (
  "HKCU\SOFTWARE\Python\PythonCore"
  "HKLM\SOFTWARE\Python\PythonCore"
  "HKLM\SOFTWARE\WOW6432Node\Python\PythonCore"
) do (
  for /f "usebackq tokens=3*" %%A in ('reg query "%%~H" /s /v ExecutablePath 2^>nul ^| findstr /i ExecutablePath') do (
    if exist "%%B" (
      set "PYTHON_EXE=%%B"
      goto :registry_found
    )
  )
)
exit /b 1

:registry_found
exit /b 0
