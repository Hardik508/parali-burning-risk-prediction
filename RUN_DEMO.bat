@echo off
title Parali burning risk prediction - demo
cd /d "%~dp0code"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%~dp0..\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
echo Using Python: %PY%
echo.
"%PY%" run_demo.py --data "..\data\real_dataset_matched.npz" --model "..\results\proposed_model.pt" --out "..\demo_output"
if errorlevel 1 goto problem
start "" "..\demo_output\demo_report.html"
if exist "..\demo_output\risk_map.html" start "" "..\demo_output\risk_map.html"
echo.
echo Done. The report and the map should now be open in your browser.
pause
exit /b 0
:problem
echo.
echo Something went wrong. If you have not done so yet, run setup_env.bat first.
pause
exit /b 1
