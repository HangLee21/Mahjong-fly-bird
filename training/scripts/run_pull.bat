@echo off
rem Pull human game data from the production backend and merge into the
rem cumulative training dataset. Runs headless for the Windows scheduled task.
setlocal
set "ROOT=%~dp0.."
set "PY=D:\MiniConda\python.exe"

if not exist "%PY%" (
    echo [ERROR] python not found at %PY% >> "%ROOT%\artifacts\pull.log"
    exit /b 1
)

"%PY%" "%~dp0pull_human_data.py" --merge-into "%ROOT%\artifacts\human_traces.jsonl" >> "%ROOT%\artifacts\pull.log" 2>&1
exit /b %ERRORLEVEL%
