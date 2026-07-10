# Modelo Predictivo para Cambio de Calderas — Hidrocasanare

## Descripción
Proyecto de grado — Especialización en Ciencia de Datos y Analítica, UNAD.

Implementación de un modelo analítico de clasificación supervisada 
(Random Forest) para apoyar la decisión operativa de cambio de 
calderas pirotubulares B801-A y B801-B en Hidrocasanare, basado en 
parámetros fisicoquímicos del agua de operación (2023-2026).

**Autor:** Manuel David Niño Mojica  
**Directora:** Lina Rocío Rivadeneira Muñoz  
**Universidad:** UNAD — 2026

## Resultados principales
- Modelo: Random Forest
- AUC-ROC prueba 2025: 0.8096
- AUC-ROC validación 2026: 0.8613
- Recall validación 2026: 0.9600
- Anticipación correcta: 10 de 12 cambios evaluables (83.3%)

## Archivos del repositorio
| Archivo | Descripción |
|---------|-------------|
| `Calderas_Hidrocasanare_OE1_OE2_V4.ipynb` | Notebook 1: Extracción de PDFs y análisis exploratorio |
| `Calderas_Hidrocasanare_Notebook2_OE2_OE3_V4.ipynb` | Notebook 2: Entrenamiento y validación del modelo |
| `Calderas_Hidrocasanare_Notebook3_Prediccion_v6.ipynb` | Notebook 3: Diagnóstico predictivo con nuevos PDFs |
| `app_calderas_v2.py` | Aplicación web Streamlit de diagnóstico operativo |
| `Ejecutar_App_Calderas.bat` | Acceso directo para ejecutar la aplicación |
| `modelo_random_forest.pkl` | Modelo Random Forest entrenado |
| `scaler.pkl` | Escalador ajustado con datos de entrenamiento |
| `medianas_imputacion.json` | Valores de imputación para nuevos datos |
| `feature_cols.json` | Orden de las 44 features del modelo |
| `parametros_modelo.json` | Métricas y constantes del modelo |

## Nota sobre los datos
Los reportes PDF del laboratorio y el dataset consolidado no están 
incluidos en este repositorio por razones de confidencialidad 
operativa de Hidrocasanare.

## Requisitos
- Python 3.9+
- Anaconda
- Ver librerías en requirements.txt
