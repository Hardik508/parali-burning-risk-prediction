@echo off
title Upload to GitHub
cd /d "%~dp0"
where git >nul 2>nul
if errorlevel 1 (
  echo Git is not installed. Install it from https://git-scm.com/download/win , then run this file again.
  pause
  exit /b 1
)
git lfs version >nul 2>nul
if errorlevel 1 (
  echo Git LFS is missing. Reinstall Git for Windows with the "Git LFS" option ticked, then run this file again.
  pause
  exit /b 1
)

for /f "delims=" %%i in ('git config --global user.name') do set GN=%%i
if not defined GN (
  set /p GN=Your name for the commit history: 
)
if not defined GN goto noname
git config --global user.name "%GN%"
for /f "delims=" %%i in ('git config --global user.email') do set GE=%%i
if not defined GE (
  set /p GE=The email of your GitHub account: 
)
if defined GE git config --global user.email "%GE%"

echo.
set /p URL=Paste the URL of your EMPTY GitHub repository (ends with .git): 
if "%URL%"=="" goto nourl

git lfs install
if not exist ".git" git init
git lfs track "*.npz"
git add .gitattributes
git add .
git commit -m "Predictive System for Parali Burning Risk Detection: code, dataset, trained model and results"
git branch -M main
git remote remove origin >nul 2>nul
git remote add origin %URL%
echo.
echo Uploading (about 115 MB, this can take several minutes). A GitHub sign-in window may open.
git push -u origin main
if errorlevel 1 (
  echo.
  echo Upload failed. Check the message above, fix it and run this file again.
  pause
  exit /b 1
)
echo.
echo Finished. Open %URL:~0,-4% in your browser.
pause
exit /b 0
:noname
echo A name is needed for the commit. Run this file again.
pause
exit /b 1
:nourl
echo No URL given. Run this file again.
pause
exit /b 1
