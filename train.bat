@echo off
SET VENV_DIR=venv

REM Check if virtual environment exists
IF NOT EXIST "%VENV_DIR%\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv %VENV_DIR%
)

REM Activate virtual environment
CALL %VENV_DIR%\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install requirements
IF EXIST requirements.txt (
    echo Installing requirements...
    pip install -r requirements.txt
) ELSE (
    echo requirements.txt not found. Skipping installation...
)

REM Run scripts
echo Running main_pretrain.py...
python main_pretrain.py

echo Running main_finetune.py...
python main_finetune.py
