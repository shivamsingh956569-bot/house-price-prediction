@echo off
title House Price Prediction App
echo Starting House Price Prediction Streamlit App...
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m streamlit run app.py
pause
