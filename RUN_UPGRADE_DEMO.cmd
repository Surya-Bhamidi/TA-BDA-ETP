@echo off
cd /d "%~dp0"
if not exist ".venv312\Scripts\python.exe" (
  echo Project Python is missing. Follow README.md setup instructions first.
  pause
  exit /b 1
)
".venv312\Scripts\python.exe" scripts\demo_project.py --file examples\upgrade_case.txt %*
set "DEMO_EXIT=%ERRORLEVEL%"
pause
exit /b %DEMO_EXIT%
