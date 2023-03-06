@echo off

setlocal
@REM call venv\Scripts\activate.bat
@REM call win32env.bat
call conda activate integrated_rgb_thermal_ortho

start "Pipeline Tool Terminal" cmd /k "python mosaic.py configs/combined.yml"
endlocal

