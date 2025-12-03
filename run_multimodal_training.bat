@echo off
REM Training script for Multimodal Dark Pattern Detection Model

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Starting multimodal model training...
python training/train_multimodal.py

pause

