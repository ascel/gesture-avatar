@echo off
echo Starting Gesture Avatar Backend Server...
echo.

cd /d "%~dp0backend"

echo Activating virtual environment (if exists)...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo Virtual environment activated.
) else (
    echo No virtual environment found. Using system Python.
)

echo.
echo Installing/updating dependencies...
pip install -r requirements.txt

echo.
echo Starting FastAPI server...
echo Backend will be available at: http://localhost:8000
echo Press Ctrl+C to stop the server
echo.

python main.py

pause 