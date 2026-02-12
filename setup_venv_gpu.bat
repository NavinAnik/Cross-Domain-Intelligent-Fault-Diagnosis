@echo off
REM Create a GPU-only venv with PyTorch (CUDA). Run from project root.
REM Usage: setup_venv_gpu.bat [cu118|cu121|cu124]
REM   cu118 = CUDA 11.8,  cu121 = CUDA 12.1,  cu124 = CUDA 12.4 (default: cu121)

set CUDA_VER=cu121
if not "%~1"=="" set CUDA_VER=%~1

set VENV_DIR=venv_gpu
set SRC=src
set PYTORCH_INDEX=https://download.pytorch.org/whl/%CUDA_VER%

echo ============================================================
echo Creating GPU venv: %VENV_DIR%  (PyTorch %CUDA_VER%)
echo ============================================================

if exist "%VENV_DIR%" (
    echo Removing existing %VENV_DIR%...
    rmdir /s /q "%VENV_DIR%"
)

echo.
echo [1/4] Creating virtual environment...
python -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo Failed to create venv. Ensure Python 3.10 or 3.11 is available.
    exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"

echo.
echo [2/4] Upgrading pip...
python -m pip install --upgrade pip -q

echo.
echo [3/4] Installing PyTorch with CUDA (%CUDA_VER%)...
pip install torch torchvision torchaudio --index-url %PYTORCH_INDEX%
if errorlevel 1 (
    echo PyTorch CUDA install failed. Try: setup_venv_gpu.bat cu118
    exit /b 1
)

echo.
echo [4/4] Installing project dependencies (no torch overwrite)...
pip install -r %SRC%\requirements_base.txt
if errorlevel 1 (
    echo Some dependencies failed. You can retry: pip install -r src\requirements_base.txt
)

echo.
echo Verifying CUDA...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo.
echo ============================================================
echo Done. Activate with:  %VENV_DIR%\Scripts\activate
echo Then run from src:   cd src  ^&  python main_dinov2.py
echo ============================================================
