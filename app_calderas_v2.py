# ══════════════════════════════════════════════════════════════════════════════
# APLICACIÓN STREAMLIT — DIAGNÓSTICO PREDICTIVO CALDERAS HIDROCASANARE
# Interfaz web para el modelo analítico de cambio de calderas
# Autor: Manuel David Niño Mojica
# ══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
import json
import re
import pdfplumber
from pathlib import Path
from scipy import stats

# ── Configuración de la página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Diagnóstico Calderas — Hidrocasanare",
    page_icon="🔥",
    layout="wide"
)

# ── Constantes de color por caldera ──────────────────────────────────────────
COLOR_A = "#1f77b4"   # azul — B801-A
COLOR_B = "#e07b39"   # naranja — B801-B

# ── Funciones auxiliares ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# CORRECCIÓN — Esta función ahora es IDÉNTICA a la de Notebook 1 (con crop)
# Antes, la app usaba page.extract_text() sobre la página completa, lo que
# mezclaba la tabla principal con el texto de las gráficas y producía valores
# faltantes (None) en variables como conductividad_purga, pH y TDS,
# especialmente en PDFs de 2025-2026 donde el layout cambió ligeramente.
# La solución es recortar primero la región de la tabla (CROP_TABLA) y
# aplicar los patrones de regex sobre ese texto limpio, igual que en Notebook 1.
# ══════════════════════════════════════════════════════════════════════════════

# ── Región de la tabla principal en todos los formatos de PDF identificados ───
CROP_TABLA = (0, 60, 420, 380)

# ── y_tolerance ajustado para evitar fusión de filas verticalmente cercanas ──
# Valor por defecto de pdfplumber: 3 (demasiado permisivo para este layout)
Y_TOLERANCE_CROP = 1

# ── Función auxiliar para convertir texto a número ────────────────────────────
def parse_float(texto):
    """
    Convierte texto del PDF a número — maneja comas, guiones de cualquier
    longitud y el caso de saturación de instrumento ">valor" (ej. ">200"),
    donde se usa el valor límite como cota inferior conservadora del dato
    real (confirmado con PDF real de 2023-03-17, sulfitos_purga = >200).
    """
    if texto is None:
        return np.nan
    texto = str(texto).strip()
    if texto in ["", "None"] or set(texto) == {"-"}:
        return np.nan
    # Caso especial: instrumento saturado, reporta ">valor"
    if texto.startswith(">"):
        texto = texto[1:].strip()
    try:
        return float(texto.replace(",", "."))
    except:
        return np.nan

# ── Función principal de extracción ──────────────────────────────────────────
def extraer_datos_pdf(ruta_pdf):
    """
    Extrae fecha, caldera activa y variables fisicoquímicas de un PDF diario.
    Usa crop de coordenadas para aislar la tabla principal y un y_tolerance
    reducido para evitar que filas verticalmente cercanas (como "Dureza Total"
    y "Óxigeno Disuelto") se fusionen y entrelacen sus caracteres.
    La fecha se toma del nombre del archivo (fuente más confiable).
    """
    try:
        with pdfplumber.open(ruta_pdf) as pdf:
            page = pdf.pages[0]

            # Texto completo para detectar caldera y fecha del encabezado
            texto_completo = page.extract_text()

            # Texto recortado para extraer valores de variables
            # y_tolerance reducido evita la fusión de filas cercanas
            region = page.crop(CROP_TABLA)
            texto  = region.extract_text(y_tolerance=Y_TOLERANCE_CROP)

            if not texto_completo:
                return None

            # ── Fecha desde el nombre del archivo ─────────────────────────────
            match_fecha = re.search(
                r"(\d{4})-(\d{2})-(\d{2})", ruta_pdf.stem)
            if not match_fecha:
                print(f"  ⚠  {ruta_pdf.name}: no se pudo extraer fecha")
                return None

            anio  = int(match_fecha.group(1))
            mes   = int(match_fecha.group(2))
            dia   = int(match_fecha.group(3))
            fecha = pd.Timestamp(year=anio, month=mes, day=dia)

            # Verificación opcional contra encabezado del PDF
            match_fecha_texto = re.search(
                r"Fecha\s+(\d{1,2})\s+"
                r"(ENERO|FEBRERO|MARZO|ABRIL|MAYO|JUNIO|"
                r"JULIO|AGOSTO|SEPTIEMBRE|OCTUBRE|NOVIEMBRE|DICIEMBRE)"
                r"\s+(\d{4})",
                texto_completo, re.IGNORECASE
            )
            if match_fecha_texto:
                anio_texto = int(match_fecha_texto.group(3))
                if anio_texto != anio:
                    print(f"  ⚠  {ruta_pdf.name}: encabezado dice año "
                          f"{anio_texto}, se usa {anio} (del nombre)")

            # ── Caldera — tres patrones en orden de prioridad ─────────────────
            # Se busca en texto_completo (sin crop, sin tocar y_tolerance)
            match_caldera = re.search(
                r"Funcionamiento\s+(B801-[AB])",
                texto_completo, re.IGNORECASE)
            if not match_caldera:
                match_caldera = re.search(
                    r"Caldera\s+en\s+(B801-[AB])",
                    texto_completo, re.IGNORECASE | re.DOTALL)
            if not match_caldera:
                match_caldera = re.search(
                    r"(B801-[AB])", texto_completo, re.IGNORECASE)
            caldera = match_caldera.group(1) if match_caldera else np.nan

            if pd.isna(caldera):
                print(f"  ⚠  {ruta_pdf.name}: caldera no detectada")

            # ── Extraer variables desde el texto recortado ────────────────────
            # pH — busca valor plausible entre 0 y 14 para entrada
            # El tercer grupo (D105) tolera guiones: confirmado con PDF real
            # (2026-04-27) que cuando D105 no tiene dato ("-"), el patrón
            # viejo (que exigía número en los 3 grupos) fallaba la línea
            # completa y arrastraba entrada/purga aunque sí tuvieran valor.
            m_ph = re.search(
                r"pH\s+((?:[0-9]|1[0-4])[,\.]\d+)\s+([\d,\.]+)\s+(-+|[\d,\.]+)",
                texto)
            if not m_ph:
                m_ph = re.search(
                    r"ENTRADA\s+PURGA\s+D105\s*\n"
                    r"(?:D105\s+\d+\s*\n)?"
                    r"pH\s+([\d,\.]+)\s+([\d,\.]+)\s+(-+|[\d,\.]+)",
                    texto)

            ph_ent  = parse_float(m_ph.group(1)) if m_ph else np.nan
            ph_purg = parse_float(m_ph.group(2)) if m_ph else np.nan
            ph_d105 = parse_float(m_ph.group(3)) if m_ph else np.nan

            if not np.isnan(ph_ent) and (ph_ent < 0 or ph_ent > 14):
                ph_ent = np.nan

            # Conductividad
            m_cond = re.search(
                r"Conductividad\s+µS/cm\s+([\d,\.]+)\s+([\d,\.]+)",
                texto)
            cond_ent  = parse_float(m_cond.group(1)) if m_cond else np.nan
            cond_purg = parse_float(m_cond.group(2)) if m_cond else np.nan

            # TDS
            m_tds = re.search(
                r"TDS\s+mg/L\s+([\d,\.]+)\s+([\d,\.]+)", texto)
            tds_ent  = parse_float(m_tds.group(1)) if m_tds else np.nan
            tds_purg = parse_float(m_tds.group(2)) if m_tds else np.nan

            # Dureza Total — acepta número o guion(es) de cualquier longitud.
            # Confirmado con PDFs reales (ene-feb 2024) que el laboratorio
            # a veces usa un solo guion "-" en vez de "-----" cuando no se
            # midió ese día — mismo patrón ya corregido en Alcalinidad.
            m_dur = re.search(
                r"Dureza\s+Total\s+mg/L\s+CaCO3\s+(-+|[\d,\.]+)", texto)
            if not m_dur:
                # Intento alternativo cuando las letras están separadas por espacios
                # (se conserva como red de seguridad para PDFs atípicos)
                m_dur = re.search(
                    r"D\s*u\s*r\s*e\s*z\s*a\s+T\s*o\s*t\s*a\s*l"
                    r"\s+mg/L\s+CaCO3\s+(-+|[\d,\.]+)", texto)
            dureza_ent = parse_float(m_dur.group(1)) if m_dur else np.nan

            # Hierro — entrada, purga, D105
            # El tercer grupo (D105) tolera guiones — mismo fix que pH,
            # confirmado con PDF real (2026-04-27)
            m_fe = re.search(
                r"Hierro\s+mg/L Fe\s+([\d,\.]+)\s+([\d,\.]+)\s+(-+|[\d,\.]+)",
                texto)
            fe_ent  = parse_float(m_fe.group(1)) if m_fe else np.nan
            fe_purg = parse_float(m_fe.group(2)) if m_fe else np.nan
            fe_d105 = parse_float(m_fe.group(3)) if m_fe else np.nan

            # Cloruros — entrada, purga, D105
            # El tercer grupo (D105) tolera guiones — mismo fix que pH,
            # confirmado con PDF real (2026-04-27)
            m_cl = re.search(
                r"Cloruros\s+mg/L Cl-?\s+([\d,\.]+)\s+([\d,\.]+)\s+(-+|[\d,\.]+)",
                texto)
            cl_ent  = parse_float(m_cl.group(1)) if m_cl else np.nan
            cl_purg = parse_float(m_cl.group(2)) if m_cl else np.nan
            cl_d105 = parse_float(m_cl.group(3)) if m_cl else np.nan

            # Alcalinidad total
            # Acepta tanto número como guion(es) de cualquier longitud (1 a 5)
            # en cada posición — algunos PDFs usan "-" simple en vez de "-----"
            # cuando el dato no fue medido (confirmado con PDFs reales de
            # marzo 2026, donde el analista usó un solo guion)
            m_alc = re.search(
                r"Alcalinidad\s+ppm CaCO3\s+(-+|[\d,\.]+)\s+(-+|[\d,\.]+)", texto)
            alc_ent  = parse_float(m_alc.group(1)) if m_alc else np.nan
            alc_purg = parse_float(m_alc.group(2)) if m_alc else np.nan

            # Alcalinidad P — mismo ajuste de tolerancia a guion simple
            m_alcp = re.search(
                r"Alcalinidad P\s+ppm CaCO3\s+-+\s+(-+|[\d,\.]+)", texto)
            alc_p_purg = parse_float(m_alcp.group(1)) if m_alcp else np.nan

            # Alcalinidad OH — mismo ajuste de tolerancia a guion simple
            m_alco = re.search(
                r"Alcalinidad OH\s+ppm CaCO3\s+-+\s+(-+|[\d,\.]+)", texto)
            alc_oh_purg = parse_float(m_alco.group(1)) if m_alco else np.nan

            # Turbidez
            m_turb = re.search(
                r"Turbidez\s+NTU\s+([\d,\.]+)\s+([\d,\.]+)", texto)
            turb_ent  = parse_float(m_turb.group(1)) if m_turb else np.nan
            turb_purg = parse_float(m_turb.group(2)) if m_turb else np.nan

            # Sílice
            m_sil = re.search(
                r"Silice\s+mg/L SO2\s+(-{1,5}|[\d,\.]+)\s+(-{1,5}|[\d,\.]+)",
                texto)
            sil_ent  = parse_float(m_sil.group(1)) if m_sil else np.nan
            sil_purg = parse_float(m_sil.group(2)) if m_sil else np.nan

            # Fosfatos — tolera el subíndice "4" en línea separada (efecto
            # secundario del y_tolerance reducido), guiones de cualquier
            # longitud cuando no hay dato, y el caso ">valor" de saturación
            # del instrumento (mismo mecanismo confirmado en Sulfitos)
            m_fos = re.search(
                r"Fosfatos\s+mg/L\s+PO\s*\n?\s*4?\s+-+\s+"
                r"(>\s*[\d,\.]+|-+|[\d,\.]+)", texto)
            fosfatos_purg = parse_float(m_fos.group(1)) if m_fos else np.nan

            # Sulfitos — mismo tratamiento que Fosfatos. Confirmado con PDF
            # real de 2023-03-17 donde el laboratorio reportó ">200" por
            # saturación del instrumento (rango operativo máx. documentado
            # es 40-50, así que >200 de cualquier forma queda fuera de rango)
            m_sul = re.search(
                r"Sulfitos\s+mg/L\s+SO\s*\n?\s*3?\s+-+\s+"
                r"(>\s*[\d,\.]+|-+|[\d,\.]+)", texto)
            sulfitos_purg = parse_float(m_sul.group(1)) if m_sul else np.nan

            # ── Construir el registro del día ─────────────────────────────────
            return {
                "fecha"                  : fecha,
                "caldera"                : caldera,
                "pH_entrada"             : ph_ent,
                "conductividad_entrada"  : cond_ent,
                "TDS_entrada"            : tds_ent,
                "dureza_entrada"         : dureza_ent,
                "hierro_entrada"         : fe_ent,
                "cloruros_entrada"       : cl_ent,
                "alcalinidad_entrada"    : alc_ent,
                "turbidez_entrada"       : turb_ent,
                "silice_entrada"         : sil_ent,
                "pH_purga"               : ph_purg,
                "conductividad_purga"    : cond_purg,
                "TDS_purga"              : tds_purg,
                "hierro_purga"           : fe_purg,
                "cloruros_purga"         : cl_purg,
                "alcalinidad_total_purga": alc_purg,
                "alcalinidad_P_purga"    : alc_p_purg,
                "alcalinidad_OH_purga"   : alc_oh_purg,
                "turbidez_purga"         : turb_purg,
                "silice_purga"           : sil_purg,
                "fosfatos_purga"         : fosfatos_purg,
                "sulfitos_purga"         : sulfitos_purg,
                "pH_D105"                : ph_d105,
                "hierro_D105"            : fe_d105,
                "cloruros_D105"          : cl_d105,
            }

    except Exception as e:
        print(f"  ⚠  Error en {ruta_pdf.name}: {e}")
        return None


@st.cache_resource
def cargar_artefactos():
    """Carga los artefactos del modelo una sola vez y los mantiene en memoria."""
    # El modelo es Random Forest entrenado con datos 2023–2025 (ver
    # Notebook 2, Sección 12 — seleccionado por mejor AUC-ROC, F1 y
    # recall simultáneamente sobre Regresión Logística)
    with open("modelo_random_forest.pkl", "rb") as f:
        modelo = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("medianas_imputacion.json", "r", encoding="utf-8") as f:
        medianas = pd.Series(json.load(f))
    with open("feature_cols.json", "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
    with open("parametros_modelo.json", "r", encoding="utf-8") as f:
        parametros = json.load(f)
    return modelo, scaler, medianas, feature_cols, parametros

def construir_features(df_input, VENTANA_CORTA, VENTANA_LARGA, UMBRAL_IDC, N_VARS):
    """Construye el vector de 44 features temporales para cada día."""
    VARS_BASE = [
        "conductividad_purga", "TDS_purga", "pH_purga",
        "fosfatos_purga", "sulfitos_purga", "alcalinidad_total_purga",
        "pH_entrada", "conductividad_entrada", "TDS_entrada",
        "alcalinidad_entrada"
    ]
    UMBRALES = {
        "pH_entrada"            : ("rango", 8.0, 9.0),
        "conductividad_entrada" : ("max", 135.0),
        "TDS_entrada"           : ("max", 67.5),
        "alcalinidad_entrada"   : ("max", 38.0),
        "pH_purga"              : ("rango", 10.5, 11.5),
        "conductividad_purga"   : ("max", 2025.0),
        "TDS_purga"             : ("max", 1013.0),
        "fosfatos_purga"        : ("rango", 25.0, 40.0),
        "sulfitos_purga"        : ("rango", 40.0, 50.0),
        "alcalinidad_total_purga": ("max", 700.0),
    }

    df_c = df_input.copy().reset_index(drop=True)
    n    = len(df_c)

    # Calcular IDC y flag_critico si no existen
    if "IDC" not in df_c.columns:
        def n_vars_fuera(row):
            n_fuera = 0
            for var, criterio in UMBRALES.items():
                if var not in row or pd.isna(row[var]):
                    continue
                val = row[var]
                if criterio[0] == "max":
                    if val > criterio[1]: n_fuera += 1
                elif criterio[0] == "rango":
                    if val < criterio[1] or val > criterio[2]: n_fuera += 1
            return n_fuera
        df_c["n_vars_riesgo"] = df_c.apply(n_vars_fuera, axis=1)
        df_c["IDC"]           = df_c["n_vars_riesgo"] / N_VARS
        df_c["flag_critico"]  = (df_c["n_vars_riesgo"] >= 5).astype(int)

    registros = []
    for i in range(VENTANA_LARGA, n):
        fila = {
            "fecha"   : df_c.loc[i, "fecha"],
            "caldera" : df_c.loc[i, "caldera"],
            "IDC_hoy" : df_c.loc[i, "IDC"],
        }
        v_c = df_c.iloc[i - VENTANA_CORTA : i]
        v_l = df_c.iloc[i - VENTANA_LARGA : i]

        for var in VARS_BASE:
            s7  = v_c[var].dropna()
            s30 = v_l[var].dropna()
            if len(s7) >= 2:
                xi        = np.arange(len(s7))
                slope, *_ = stats.linregress(xi, s7)
                fila[f"{var}_media7"] = s7.mean()
                fila[f"{var}_std7"]   = s7.std()
                fila[f"{var}_slope7"] = slope
            else:
                fila[f"{var}_media7"] = np.nan
                fila[f"{var}_std7"]   = np.nan
                fila[f"{var}_slope7"] = np.nan
            fila[f"{var}_media30"] = s30.mean() if len(s30) >= 2 else np.nan

        arr_idc   = (v_c["IDC"] >= UMBRAL_IDC).values
        racha_idc = 0
        for val in reversed(arr_idc):
            if val: racha_idc += 1
            else: break
        fila["dias_consec_critico"] = racha_idc
        fila["IDC_media7"]  = v_c["IDC"].mean()
        fila["IDC_media30"] = v_l["IDC"].mean()
        fila["IDC_slope7"]  = np.nan

        s_idc = v_c["IDC"].dropna()
        if len(s_idc) >= 2:
            xi_idc        = np.arange(len(s_idc))
            slope_idc, *_ = stats.linregress(xi_idc, s_idc)
            fila["IDC_slope7"] = slope_idc

        fila["y"] = int(df_c.loc[i, "flag_critico"])
        registros.append(fila)

    return pd.DataFrame(registros)

def dias_consecutivos_alerta(df_cal, umbral=0.5):
    """Cuenta días consecutivos en alerta hasta el último registro."""
    serie = df_cal.sort_values(
        "fecha", ascending=False)["prob_critico"].values
    racha = 0
    for p in serie:
        if p >= umbral: racha += 1
        else: break
    return racha

def fecha_inicio_episodio_actual(caldera, cambios_lista, fecha_min_datos):
    """
    Encuentra la fecha en que la caldera entró en operación por última vez,
    usando la lista de cambios detectados. Evita mezclar la racha de alerta
    actual con un episodio anterior de la misma caldera (bug confirmado y
    corregido en Notebook 3: si una caldera operó, salió, y reentró más
    tarde, la racha solo debe contar desde que reentró).
    """
    entradas = [c["fecha"] for c in cambios_lista
                if c["caldera_entra"] == caldera]
    if entradas:
        return max(entradas)
    return fecha_min_datos

def insertar_cortes_huecos(df_cal, umbral_dias=2):
    """
    Detecta huecos en la serie temporal de una caldera (días sin registro,
    porque estuvo fuera de servicio / en stand-by) e inserta una fila con
    prob_critico=NaN justo después del último día antes del hueco.
    matplotlib no dibuja línea sobre un NaN, así que esto corta visualmente
    la línea en vez de unir directamente el último día de un episodio con
    el primero del siguiente (lo cual se veía como operación continua
    cuando en realidad la caldera estuvo apagada en el medio).
    Misma lógica que en el Notebook 3, portada aquí para que la app
    muestre el mismo corte visual.

    Parámetros:
        df_cal      : DataFrame de una sola caldera, ordenado por fecha
        umbral_dias : si el salto entre dos fechas consecutivas supera este
                      número de días, se considera un hueco de operación
    Retorna:
        DataFrame con filas NaN insertadas en los huecos detectados
    """
    df_cal = df_cal.sort_values("fecha").reset_index(drop=True)
    filas_corte = []

    for i in range(1, len(df_cal)):
        salto = (df_cal.loc[i, "fecha"] - df_cal.loc[i-1, "fecha"]).days
        if salto > umbral_dias:
            fila_nan = df_cal.loc[i-1].copy()
            fila_nan["fecha"] = df_cal.loc[i-1, "fecha"] + pd.Timedelta(days=1)
            fila_nan["prob_critico"] = np.nan
            fila_nan["IDC_hoy"] = np.nan
            filas_corte.append(fila_nan)

    if filas_corte:
        df_cal = pd.concat(
            [df_cal, pd.DataFrame(filas_corte)], ignore_index=True
        ).sort_values("fecha").reset_index(drop=True)

    return df_cal

# ══════════════════════════════════════════════════════════════════════════════
# INTERFAZ STREAMLIT
# ══════════════════════════════════════════════════════════════════════════════

# ── Encabezado ────────────────────────────────────────────────────────────────
st.title("🔥 Diagnóstico Predictivo — Calderas Hidrocasanare")
st.markdown(
    "**Modelo:** Random Forest entrenado con datos 2023–2025 | "
    "**OE3/OE4** — Proyecto de grado Especialización Ciencia de Datos UNAD"
)
st.divider()

# ── Barra lateral — configuración ────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")

    # Ruta de datos nuevos
    ruta_nuevos = st.text_input(
        "Ruta carpeta datos nuevos (PDFs):",
        value=r"F:\Cursos\Unad\Posgrado\(2026) PERIODO 16-02\PROYECTO DE GRADO II\Datos\datos_nuevos",
        help="Carpeta raíz que contiene subcarpetas por año y mes con los PDFs"
    )

    # Botón para procesar
    procesar = st.button(
        "🔄 Actualizar diagnóstico",
        type="primary",
        use_container_width=True
    )

    st.divider()
    st.markdown("**Criterios de alerta:**")
    st.markdown("🟢 **CONTROLADO:** prob < 50 %")
    st.markdown("🟡 **ALERTA:** prob 50–75 %")
    st.markdown("🔴 **CRÍTICO:** prob > 75 %")
    st.markdown("⭐ **CAMBIO REC.:** prob ≥ 65 %, ≥3d alerta, IDC↑")

# ── Cargar artefactos ─────────────────────────────────────────────────────────
try:
    modelo, scaler, medianas, feature_cols, parametros = cargar_artefactos()
    VENTANA_CORTA = parametros["ventana_corta_dias"]
    VENTANA_LARGA = parametros["ventana_larga_dias"]
    UMBRAL_IDC    = parametros["umbral_idc"]
    N_VARS        = parametros["n_vars_monitoreadas"]
    st.sidebar.success(f"✔ Modelo cargado | AUC: {parametros['auc_roc_prueba']}")
except Exception as e:
    st.error(f"Error cargando artefactos del modelo: {e}")
    st.stop()

# ── Procesamiento al hacer clic ───────────────────────────────────────────────
if procesar:

    with st.spinner("Extrayendo PDFs nuevos..."):

        ruta_raiz = Path(ruta_nuevos)
        if not ruta_raiz.exists():
            st.error(f"Carpeta no encontrada: {ruta_raiz}")
            st.stop()

        # Extraer todos los PDFs recursivamente
        todos_pdfs   = sorted(ruta_raiz.rglob("*.pdf"))
        registros    = []
        errores      = []

        progress_bar = st.progress(0)
        for idx, pdf_path in enumerate(todos_pdfs):
            registro = extraer_datos_pdf(pdf_path)
            if registro is not None:
                registros.append(registro)
            else:
                errores.append(pdf_path.name)
            progress_bar.progress((idx + 1) / len(todos_pdfs))

        if not registros:
            st.error("No se pudo extraer ningún registro de los PDFs")
            st.stop()

        df_nuevos = pd.DataFrame(registros)
        df_nuevos["fecha"] = pd.to_datetime(df_nuevos["fecha"])
        df_nuevos = df_nuevos.sort_values("fecha").reset_index(drop=True)

        if errores:
            st.warning(f"⚠ {len(errores)} PDFs con error: {', '.join(errores[:5])}")

    with st.spinner("Construyendo features y generando predicciones..."):

        # Cargar histórico
        df_hist = pd.read_csv("dataset_con_etiqueta.csv", encoding="utf-8-sig")
        df_hist["fecha"] = pd.to_datetime(df_hist["fecha"])

        # Combinar histórico con nuevos
        df_combinado = pd.concat(
            [df_hist, df_nuevos], ignore_index=True
        ).sort_values("fecha").drop_duplicates(
            subset=["fecha", "caldera"]
        ).reset_index(drop=True)

        # Construir features
        df_feat = construir_features(
            df_combinado, VENTANA_CORTA, VENTANA_LARGA, UMBRAL_IDC, N_VARS)
        df_feat = df_feat.dropna(subset=["y"]).reset_index(drop=True)

        # Filtrar solo nuevos
        fecha_inicio = df_nuevos["fecha"].min()
        df_pred = df_feat[
            df_feat["fecha"] >= fecha_inicio
        ].copy().reset_index(drop=True)

        # Predecir
        X_nuevo = df_pred[feature_cols].fillna(medianas)
        # IMPORTANTE: Random Forest se entrenó SIN escalar (Notebook 2,
        # Sección 9.3, rama "else" para Árbol/Random Forest) — los árboles
        # dividen el espacio de features por umbrales absolutos, no son
        # sensibles a la escala. Se predice con X_nuevo (sin escalar), no
        # con X_nuevo_sc. Usar la versión escalada produce predicciones
        # degeneradas (mismo bug ya corregido en Notebook 2 y Notebook 3).
        X_nuevo_sc = scaler.transform(X_nuevo)   # se mantiene por si se usa Reg. Log. en el futuro
        df_pred["prob_critico"] = modelo.predict_proba(X_nuevo)[:, 1]
        df_pred["pred_critico"] = modelo.predict(X_nuevo)

    # ── Mostrar resultados ────────────────────────────────────────────────────
    st.success(
        f"✔ {len(registros)} registros procesados | "
        f"Período: {df_nuevos['fecha'].min().date()} → "
        f"{df_nuevos['fecha'].max().date()}"
    )

    # Detectar cambios de caldera
    df_sorted = df_pred.sort_values("fecha").reset_index(drop=True)
    cambios   = []
    for i in range(1, len(df_sorted)):
        if df_sorted.loc[i, "caldera"] != df_sorted.loc[i-1, "caldera"]:
            cambios.append({
                "fecha"        : df_sorted.loc[i, "fecha"],
                "caldera_entra": df_sorted.loc[i,   "caldera"],
                "caldera_sale" : df_sorted.loc[i-1, "caldera"],
            })

    # ── Semáforo por caldera ──────────────────────────────────────────────────
    st.subheader("🚦 Estado actual de calderas")
    cols = st.columns(2)

    resultados = {}

    for col, caldera in zip(cols, ["B801-A", "B801-B"]):
        df_cal = df_pred[
            df_pred["caldera"] == caldera
        ].sort_values("fecha").reset_index(drop=True)

        with col:
            if df_cal.empty:
                st.warning(f"{caldera}: sin datos en el período")
                continue

            ultima     = df_cal.iloc[-1]
            prob       = ultima["prob_critico"]
            idc_rec    = ultima["IDC_hoy"]
            idc_slope  = ultima["IDC_slope7"] if not pd.isna(
                ultima["IDC_slope7"]) else 0
            fecha_eval = ultima["fecha"].date()

            # Acotar al episodio de operación MÁS RECIENTE de esta caldera
            # (evita mezclar la racha actual con un episodio anterior)
            fecha_inicio_actual = fecha_inicio_episodio_actual(
                caldera, cambios, df_pred["fecha"].min()
            )
            df_cal_episodio_actual = df_cal[
                df_cal["fecha"] >= fecha_inicio_actual
            ].copy()
            dias_alerta = dias_consecutivos_alerta(df_cal_episodio_actual)

            # Clasificación
            rec_cambio = (prob >= 0.65 and
                         dias_alerta >= 3 and
                         idc_slope > 0)

            if rec_cambio:
                estado = "⭐ CAMBIO RECOMENDADO"
                color  = "red"
            elif prob >= 0.75 or (idc_rec >= UMBRAL_IDC and idc_slope > 0):
                estado = "🔴 CRÍTICO"
                color  = "red"
            elif prob >= 0.50 or idc_rec >= UMBRAL_IDC:
                estado = "🟡 ALERTA"
                color  = "orange"
            else:
                estado = "🟢 CONTROLADO"
                color  = "green"

            resultados[caldera] = {
                "estado"     : estado,
                "prob"       : prob,
                "idc_rec"    : idc_rec,
                "idc_slope"  : idc_slope,
                "dias_alerta": dias_alerta,
                "fecha_eval" : fecha_eval,
                "df_cal"     : df_cal,
                "rec_cambio" : rec_cambio,
                "color"      : color,
            }

            # Tarjeta de diagnóstico
            st.markdown(f"### {caldera}")
            st.markdown(
                f"<h2 style='color:{color}'>{estado}</h2>",
                unsafe_allow_html=True
            )
            st.metric("Probabilidad crítico", f"{prob*100:.1f} %")
            st.metric("IDC reciente (7d)",
                      f"{idc_rec:.3f}",
                      delta=f"{idc_slope:+.4f} ↑ deterioro"
                      if idc_slope > 0 else f"{idc_slope:+.4f} ↓ mejora")
            st.metric("Días consecutivos en alerta", f"{dias_alerta}")
            st.caption(f"Última evaluación: {fecha_eval}")

    st.divider()

    # ── Gráfica de probabilidad ───────────────────────────────────────────────
    st.subheader("📈 Evolución de probabilidad predicha")

    colores_cal = {"B801-A": COLOR_A, "B801-B": COLOR_B}
    n_cal       = len(resultados)

    fig, axes = plt.subplots(n_cal, 1,
                              figsize=(14, 4 * n_cal),
                              sharex=True)
    if n_cal == 1:
        axes = [axes]

    for ax, (caldera, res) in zip(axes, resultados.items()):
        # Insertar cortes visuales en los huecos de operación (caldera
        # apagada/standby), igual que en el Notebook 3, para que la línea
        # no aparente continuidad falsa entre episodios separados
        df_cal = insertar_cortes_huecos(res["df_cal"])
        color  = colores_cal.get(caldera, "#333333")

        ax.set_facecolor(
            "#fff0f0" if "CRÍTICO" in res["estado"] or
            "CAMBIO" in res["estado"]
            else "#f0fff0" if "CONTROLADO" in res["estado"]
            else "#fffdf0"
        )

        # Zonas
        ax.axhspan(0.75, 1.05, alpha=0.05, color="red")
        ax.axhspan(0.50, 0.75, alpha=0.05, color="orange")

        # Probabilidad
        ax.plot(df_cal["fecha"], df_cal["prob_critico"],
                color=color, linewidth=2.0, alpha=0.85,
                label="Prob. crítico")
        ax.fill_between(df_cal["fecha"], df_cal["prob_critico"],
                        alpha=0.12, color=color)

        # IDC real
        ax.plot(df_cal["fecha"], df_cal["IDC_hoy"],
                color="gray", linewidth=1.0, alpha=0.5,
                linestyle="--", label="IDC real")

        # Umbrales
        ax.axhline(0.5, color="red", linestyle="--",
                   linewidth=1.2, alpha=0.7, label="Umbral (0,5)")
        ax.axhline(0.65, color="#8B0000", linestyle=":",
                   linewidth=1.0, alpha=0.6, label="Cambio rec. (0,65)")

        # Cambios reales
        for cambio in cambios:
            if cambio["caldera_sale"] == caldera:
                ax.axvline(cambio["fecha"], color="black",
                           linewidth=2.0, alpha=0.85)
                ax.text(cambio["fecha"], 0.96,
                        f"Cambio\n{cambio['fecha'].strftime('%d/%m')}",
                        ha="center", va="top", fontsize=7,
                        fontweight="bold",
                        transform=ax.get_xaxis_transform(),
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="white",
                                  edgecolor="black",
                                  linewidth=0.8, alpha=0.9))

        # Estrellas de cambio recomendado
        df_cal_s = df_cal.sort_values("fecha").reset_index(drop=True)
        fechas_estrella = []
        for i in range(len(df_cal_s)):
            sub   = df_cal_s.iloc[:i+1]
            racha = 0
            for p in sub["prob_critico"].values[::-1]:
                if p >= 0.5: racha += 1
                else: break
            prob_i  = df_cal_s.loc[i, "prob_critico"]
            slope_i = df_cal_s.loc[i, "IDC_slope7"]
            if pd.isna(slope_i): slope_i = 0
            if prob_i >= 0.65 and racha >= 3 and slope_i > 0:
                fechas_estrella.append(df_cal_s.loc[i, "fecha"])

        if fechas_estrella:
            probs_e = df_cal[
                df_cal["fecha"].isin(fechas_estrella)
            ]["prob_critico"].values
            ax.scatter(fechas_estrella, probs_e,
                       marker="*", color="#8B0000",
                       s=200, zorder=6,
                       label=f"Cambio rec. ({len(fechas_estrella)}d)")

        # Punto final
        ax.scatter(
            [df_cal["fecha"].iloc[-1]],
            [df_cal["prob_critico"].iloc[-1]],
            color="#8B0000" if res["rec_cambio"] else color,
            marker="*" if res["rec_cambio"] else "o",
            s=300 if res["rec_cambio"] else 150,
            zorder=7,
            label=f"Hoy: {res['estado']}"
        )

        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Probabilidad / IDC", fontsize=9)
        ax.set_title(
            f"{caldera} — {res['estado']} | "
            f"Prob.: {res['prob']*100:.1f}% | "
            f"IDC: {res['idc_rec']:.2f} | "
            f"Días alerta: {res['dias_alerta']}",
            fontsize=9, fontweight="bold"
        )
        ax.legend(fontsize=7.5, loc="lower left",
                  ncol=3, framealpha=0.9)
        ax.grid(axis="y", alpha=0.25)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d\n%b"))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axes[-1].set_xlabel("Fecha", fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    # ── Tabla de cambios detectados ───────────────────────────────────────────
    if cambios:
        st.subheader("🔄 Cambios de caldera detectados")
        df_cambios = pd.DataFrame(cambios)
        df_cambios["fecha"] = df_cambios["fecha"].dt.date
        st.dataframe(df_cambios, use_container_width=True)

    # ── Tabla de datos extraídos ──────────────────────────────────────────────
    with st.expander("📋 Ver datos extraídos de PDFs"):
        st.dataframe(
            df_nuevos[["fecha", "caldera", "pH_entrada",
                       "conductividad_purga", "fosfatos_purga",
                       "sulfitos_purga", "pH_purga"]],
            use_container_width=True
        )

else:
    # Pantalla de bienvenida antes de procesar
    st.info(
        "👈 Configura la ruta de los PDFs en el panel izquierdo "
        "y haz clic en **Actualizar diagnóstico** para generar el análisis."
    )

    st.markdown("### Como usar esta aplicación?")
    st.markdown(
        "1. **Verifica la ruta** de la carpeta con los PDFs en el panel izquierdo\n"
        "2. **Haz clic** en *Actualizar diagnostico*\n"
        "3. La aplicacion extrae los PDFs, construye las features y genera el semaforo\n"
        "4. Cuando llegue un nuevo mes: agrega los PDFs a su carpeta y vuelve a hacer clic"
    )

    st.markdown("### Estructura esperada de carpetas:")
    st.code(
        "datos_nuevos/\n"
        "2026/\n"
        "  5. Mayo/\n"
        "    Monitoreo de aguas 2026-05-01.pdf\n"
        "  6. Junio/\n"
        "2027/\n"
        "  1. Enero/",
        language=None
    )

    st.markdown("### Modelo:")
    st.markdown(
        "- **Algoritmo:** Regresion Logistica\n"
        "- **Entrenado con:** datos 2023-2025\n"
        "- **AUC-ROC prueba:** 0,7366\n"
        "- **AUC-ROC validacion 2026:** 0,7494"
    )