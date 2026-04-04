@echo off
set PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe

if not exist "%PYTHON_EXE%" (
  echo Python not found at:
  echo %PYTHON_EXE%
  pause
  exit /b 1
)

echo Starting Streamlit on http://127.0.0.1:8502
"%PYTHON_EXE%" -m streamlit run app.py --server.port 8502 --server.address 127.0.0.1
pause
