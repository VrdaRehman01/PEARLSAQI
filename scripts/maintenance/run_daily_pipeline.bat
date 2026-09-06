@echo off
setlocal EnableExtensions

cd /d "C:\Users\warda\Downloads\aqi-predictor-v4"

echo ============================================================
echo PEARLSAQI DAILY PIPELINE
echo ============================================================
echo.

set "PYTHON=C:\Users\warda\Downloads\aqi-predictor-v4\venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo ERROR: Virtual environment Python not found:
    echo %PYTHON%
    exit /b 1
)

echo Using Python:
echo %PYTHON%
echo Working Directory:
cd
echo.

echo ============================================================
echo STEP 1 - DAILY AQI / WEATHER / FORECAST PIPELINE
echo ============================================================
echo.

"%PYTHON%" -m src.pipelines.daily_update

if errorlevel 1 (
    echo.
    echo ============================================================
    echo DAILY PIPELINE FAILED
    echo ============================================================
    echo ERRORLEVEL=%ERRORLEVEL%
    exit /b 1
)

echo.
echo DAILY PIPELINE COMPLETED SUCCESSFULLY.
echo.

echo ============================================================
echo STEP 2 - MODEL VS BASELINE
echo ============================================================
echo.

"%PYTHON%" compare_production_baseline.py

if errorlevel 1 (
    echo.
    echo BASELINE COMPARISON FAILED
    echo ============================================================
    exit /b 1
)

echo.
echo BASELINE COMPARISON COMPLETED.
echo.

echo ============================================================
echo STEP 3 - PRODUCTION MONITOR
echo ============================================================
echo.

"%PYTHON%" inspect_production_monitor.py

if errorlevel 1 (
    echo.
    echo PRODUCTION MONITOR FAILED
    echo ============================================================
    exit /b 1
)

echo.
echo PRODUCTION MONITOR COMPLETED.
echo.

echo ============================================================
echo STEP 4 - MODEL DRIFT MONITOR
echo ============================================================
echo.

"%PYTHON%" inspect_model_drift.py

if errorlevel 1 (
    echo.
    echo MODEL DRIFT MONITOR FAILED
    echo ============================================================
    exit /b 1
)

echo.
echo MODEL DRIFT MONITOR COMPLETED.
echo.

echo ============================================================
echo STEP 5 - FINAL SYSTEM HEALTH CHECK
echo ============================================================
echo.

"%PYTHON%" inspect_daily_system.py

if errorlevel 1 (
    echo.
    echo HEALTH CHECK FAILED
    echo ============================================================
    exit /b 1
)

echo.
echo ============================================================
echo ALL DAILY CHECKS PASSED
echo ============================================================
echo.

exit /b 0