@echo off
title Diagnóstico Predictivo de Calderas — Hidrocasanare
color 1F

echo.
echo  ================================================================
echo   SISTEMA DE DIAGNÓSTICO PREDICTIVO DE CALDERAS — HIDROCASANARE
echo   Especialización en Ciencia de Datos y Analítica — UNAD
echo  ================================================================
echo.
echo  Iniciando aplicación...
echo  Por favor espere mientras carga el modelo y los datos.
echo.
echo  Una vez iniciada, la aplicación se abrirá automáticamente
echo  en su navegador en la dirección: http://localhost:8501
echo.
echo  Para detener la aplicación cierre esta ventana o presione Ctrl+C
echo.
echo  ================================================================
echo.

cd /d "F:\Cursos\Unad\Posgrado\(2026) PERIODO 16-02\PROYECTO DE GRADO II\Datos"

"c:\ProgramData\anaconda3\python.exe" -m streamlit run app_calderas_v2.py --server.headless false --browser.gatherUsageStats false

echo.
echo  La aplicación se ha detenido.
pause
