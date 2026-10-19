@echo off
title Setup: Parali burning risk prototype
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10 or newer from https://www.python.org and tick "Add to PATH".
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment .venv ...
  python -m venv .venv
)
echo Installing packages (this can take a few minutes) ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Package installation failed. Check the messages above.
  pause
  exit /b 1
)
echo.
echo Setup finished. Double-click RUN_DEMO.bat to run the prototype.
pause
