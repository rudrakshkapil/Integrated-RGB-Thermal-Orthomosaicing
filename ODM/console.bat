@echo off

call conda deactivate

setlocal
@REM call venv\Scripts\activate.bat
call win32env.bat

start "ODM Console" cmd /k "echo  ____________________________ && echo /   ____    _____    __  __  \ && echo ^|  / __ \  ^|  __ \  ^|  \/  ^| ^| && echo ^| ^| ^|  ^| ^| ^| ^|  ^| ^| ^| \  / ^| ^| && echo ^| ^| ^|  ^| ^| ^| ^|  ^| ^| ^| ^|\/^| ^| ^| && echo ^| ^| ^|__^| ^| ^| ^|__^| ^| ^| ^|  ^| ^| ^| && echo ^|  \____/  ^|_____/  ^|_^|  ^|_^| ^| && echo \____________________________/ && @echo off && FOR /F %%i in (VERSION) do echo        version: %%i && @echo on && echo. && run --help

endlocal

call conda activate integrated_rgb_thermal_ortho
