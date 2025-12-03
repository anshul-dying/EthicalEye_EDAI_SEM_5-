@echo off
echo Starting Ethical Eye API...
echo.
echo Checking for virtual environment...
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

echo.
echo Installing/Verifying dependencies...
pip install -r requirements.txt

echo.
echo Starting API server...
python api/ethical_eye_api.py
pause
