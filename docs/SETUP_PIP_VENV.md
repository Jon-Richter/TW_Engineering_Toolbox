# Setup with pip + venv (developer only)

This is intended for developers. End users should use vendor mode.
**Use Python 3.13**; QtWebEngine is not available, so web tools open in your default browser.
Capture/export buttons from embedded web tools are not available in browser mode.

## Create venv
From toolbox root:
- `python -m venv .venv`

Activate:
- Command Prompt: `.venv\Scripts\activate.bat`
- PowerShell: `.venv\Scripts\Activate.ps1`

## Install dependencies
- `pip install --upgrade pip`
- `pip install -r requirements.bundle.txt`

`requirements.bundle.txt` includes core UI deps and tool-specific packages.

## Run
- `python -m toolbox_app`
