@echo off
echo Installing required packages...
pip install flask pandas scikit-learn
echo Installing required packages complete.
start cmd /k python app.py
pause