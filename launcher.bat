@echo off

setlocal
@REM call venv\Scripts\activate.bat
@REM call win32env.bat
call conda activate cynthia_clone

start "Pipeline Tool Terminal" cmd /k "python mosaic.py configs/combined.yml"
endlocal

