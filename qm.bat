@echo off
REM ===========================================================================
REM  LCHQMDriver launcher (Windows)
REM  Usage:  qm.bat start   -> launch the QUAlibrate server (default)
REM          qm.bat setup   -> run the interactive QUAlibrate configuration
REM          double-click   -> same as "start"
REM ===========================================================================

REM --- Environment: uv venv .venv-qm (see SCQO/INSTALL.md section 1); ----------
REM     Resolved RELATIVE to this repo on purpose. INSTALL.md's one hard
REM     requirement is the SHAPE -- the repos sit as siblings under one parent --
REM     so the shared venv is always <parent>\.venv-qm, whatever that parent is
REM     called on a given machine (D:\github on the lab PC, elsewhere on others).
REM     conda LCHQM_test is the legacy fallback until the venv is battle-tested.
set "VENV_ACTIVATE=%~dp0..\.venv-qm\Scripts\activate.bat"
set "ENV_NAME=LCHQM_test"
REM ----------------------------------------------------------------------------

REM Run from the folder this script lives in (so relative config paths resolve)
cd /d %~dp0

REM Default action is "start" when no argument / double-clicked
set "CMD=%~1"
if "%CMD%"=="" set "CMD=start"

REM "qm.bat conda ..." forces the legacy conda env (fallback while transitioning)
if /I "%CMD%"=="conda" (
    set "CMD=%~2"
    if "%CMD%"=="" set "CMD=start"
    goto use_conda
)

if exist "%VENV_ACTIVATE%" (
    echo Activating venv .venv-qm ...
    CALL "%VENV_ACTIVATE%"
    goto env_ready
)

:use_conda
REM Locate conda's activate.bat (miniconda3, then anaconda3)
set "ACTIVATE_PATH=%USERPROFILE%\miniconda3\Scripts\activate.bat"
if not exist "%ACTIVATE_PATH%" set "ACTIVATE_PATH=%USERPROFILE%\anaconda3\Scripts\activate.bat"
if not exist "%ACTIVATE_PATH%" (
    echo [ERROR] Could not find conda activate.bat under:
    echo         %USERPROFILE%\miniconda3\Scripts\activate.bat
    echo         %USERPROFILE%\anaconda3\Scripts\activate.bat
    echo Edit ACTIVATE_PATH in this script to point at your conda installation.
    pause
    exit /b 1
)

echo Activating conda environment "%ENV_NAME%" ...
CALL "%ACTIVATE_PATH%" %ENV_NAME%

:env_ready

if /I "%CMD%"=="setup" (
    setup-qualibrate-config
) else if /I "%CMD%"=="start" (
    qualibrate start
) else (
    echo [ERROR] Unknown command "%CMD%".
    echo Usage: qm.bat [start^|setup]
)

REM Keep the window open so output / errors stay visible after a double-click
pause
