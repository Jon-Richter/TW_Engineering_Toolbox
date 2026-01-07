# End-user setup (read-only SharePoint/Teams folder)

Assumptions:
- The toolbox code is in a **synced** SharePoint/Teams folder (OneDrive sync) on the user's machine.
- The SharePoint folder is **read-only** for normal users.
- Each user has **Python 3.13** installed (no other deps required).
  - Note: Qt WebEngine is not available on 3.13, so web tools open in the default browser.
    Capture/export buttons in the host UI are not available in browser mode.

## 1) Sync the toolbox folder
1. Open the Teams channel > Files tab, click **Open in SharePoint**
2. Click **Sync** (OneDrive)
3. Confirm the folder appears on your PC (File Explorer) and you can see:
   - `toolbox_app/`
   - `scripts/`
   - `README.md`

Important: running tools from the web UI will not work; it must be the synced local path.

## 2) Use vendor mode (no installs)
Ship a prebuilt `vendor/` folder so users only need Python.

On a build machine with internet:
1. Run `scripts\vendorize_deps.bat`
2. Distribute the repo **including** the `vendor/` folder

On end-user machines:
- Run `Launch Toolbox.bat`

Optional: set `TOOLBOX_PYTHON_EXE` if `python` is not on PATH.
Optional: set `TOOLBOX_USE_VENDOR=0` to use locally installed site-packages instead of `vendor/`.
If you prefer local packages, run `install_deps.bat` once to install `requirements.txt`.

## 3) Run the toolbox
Re-run `Launch Toolbox.bat`.

## Outputs and logs location
This template writes logs and outputs to a user-writable folder:
- `%LOCALAPPDATA%\EngineeringToolbox\`

This avoids SharePoint permission/locking issues.

## Offline deployment (no internet needed)
Vendor mode is offline by design once you ship `vendor/`.
