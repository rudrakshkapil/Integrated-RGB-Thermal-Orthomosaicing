@echo off
rem Bypass "Terminate Batch Job" prompt.

call conda deactivate
call conda deactivate 
call conda deactivate

setlocal

call conda deactivate

cd /d %~dp0
@REM venv\Scripts\activate.bat
winrun.bat %* <NUL

@REM conda activate cynthia



endlocal

@REM TODO: generalize
call conda activate integrated_rgb_thermal_ortho