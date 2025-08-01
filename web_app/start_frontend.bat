@echo off
echo Starting Gesture Avatar Frontend...
echo.

cd /d "%~dp0frontend"

echo Installing/updating Node.js dependencies...
npm install

echo.
echo Starting React development server...
echo Frontend will be available at: http://localhost:3000
echo Press Ctrl+C to stop the server
echo.

npm start

pause 