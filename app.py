import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import google.generativeai as genai

# Configuración de página
st.set_page_config(page_title="Prueba de Hipótesis", layout="wide")
st.title("App de Prueba de Hipótesis Z")

# ── MÓDULO 1: CARGA DE DATOS ──────────────────────────────────────
st.header("1. Carga de Datos")
opcion = st.radio("¿Cómo quieres ingresar los datos?", ["Generar datos sintéticos", "Subir archivo CSV"])

if opcion == "Generar datos sintéticos":
    media_real = st.slider("Media real de los datos", 0.0, 100.0, 50.0)
    desv_real  = st.slider("Desviación estándar", 1.0, 30.0, 10.0)
    n_datos    = st.slider("Número de observaciones", 30, 500, 100)
    np.random.seed(42)
    datos = np.random.normal(media_real, desv_real, n_datos)
    st.success(f"Se generaron {n_datos} datos sintéticos.")
else:
    archivo = st.file_uploader("Sube tu CSV", type=["csv"])
    if archivo:
        df = pd.read_csv(archivo)
        st.dataframe(df.head())
        columna = st.selectbox("Selecciona la variable numérica", df.select_dtypes(include=np.number).columns)
        datos = df[columna].dropna().values
    else:
        st.warning("Sube un archivo CSV para continuar.")
        st.stop()

# ── MÓDULO 2: VISUALIZACIÓN ───────────────────────────────────────
st.header("2. Visualización de Distribución")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    ax1.hist(datos, bins=20, color="steelblue", edgecolor="white", density=True)
    xmin, xmax = ax1.get_xlim()
    x = np.linspace(xmin, xmax, 200)
    ax1.plot(x, stats.norm.pdf(x, np.mean(datos), np.std(datos)), "r-", linewidth=2, label="KDE normal")
    ax1.set_title("Histograma con curva normal")
    ax1.legend()
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    ax2.boxplot(datos, vert=True, patch_artist=True,
                boxprops=dict(facecolor="steelblue", color="navy"))
    ax2.set_title("Boxplot")
    st.pyplot(fig2)

# Análisis automático
sesgo = stats.skew(datos)
_, p_norm = stats.shapiro(datos[:50] if len(datos) > 50 else datos)
st.subheader("Análisis de la distribución")
st.write(f"**Sesgo:** {sesgo:.3f} → {'sin sesgo notable' if abs(sesgo)<0.5 else 'con sesgo'}")
st.write(f"**¿Parece normal?** {'Sí (p={:.3f})'.format(p_norm) if p_norm > 0.05 else 'No exactamente (p={:.3f})'.format(p_norm)}")
q1, q3 = np.percentile(datos, [25, 75])
iqr = q3 - q1
outliers = np.sum((datos < q1 - 1.5*iqr) | (datos > q3 + 1.5*iqr))
st.write(f"**Outliers detectados:** {outliers}")

# ── MÓDULO 3: PRUEBA Z ────────────────────────────────────────────
st.header("3. Prueba de Hipótesis Z")

col3, col4 = st.columns(2)
with col3:
    mu0   = st.number_input("Hipótesis nula H₀: µ =", value=50.0)
    sigma = st.number_input("Desviación estándar poblacional (σ)", value=10.0, min_value=0.1)
    alpha = st.selectbox("Nivel de significancia (α)", [0.01, 0.05, 0.10])
with col4:
    tipo_prueba = st.radio("Tipo de prueba", ["Bilateral (≠)", "Cola izquierda (<)", "Cola derecha (>)"])

if st.button(" Ejecutar prueba Z"):
    n        = len(datos)
    x_barra  = np.mean(datos)
    Z        = (x_barra - mu0) / (sigma / np.sqrt(n))

    if tipo_prueba == "Bilateral (≠)":
        p_value  = 2 * (1 - stats.norm.cdf(abs(Z)))
        z_critico = stats.norm.ppf(1 - alpha/2)
        rechazar  = abs(Z) > z_critico
    elif tipo_prueba == "Cola izquierda (<)":
        p_value  = stats.norm.cdf(Z)
        z_critico = -stats.norm.ppf(1 - alpha)
        rechazar  = Z < z_critico
    else:
        p_value  = 1 - stats.norm.cdf(Z)
        z_critico = stats.norm.ppf(1 - alpha)
        rechazar  = Z > z_critico

    # Resultados
    st.subheader("Resultados")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Media muestral", f"{x_barra:.4f}")
    c2.metric("Tamaño n", n)
    c3.metric("Estadístico Z", f"{Z:.4f}")
    c4.metric("p-value", f"{p_value:.4f}")

    if rechazar:
        st.error(f"Se RECHAZA H₀ (p={p_value:.4f} < α={alpha})")
    else:
        st.success(f" No se rechaza H₀ (p={p_value:.4f} ≥ α={alpha})")

    # Curva
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    x_vals = np.linspace(-4, 4, 300)
    ax3.plot(x_vals, stats.norm.pdf(x_vals), "k-", linewidth=2)

    if tipo_prueba == "Bilateral (≠)":
        ax3.fill_between(x_vals, stats.norm.pdf(x_vals),
                         where=(x_vals <= -z_critico), color="red", alpha=0.4, label="Zona de rechazo")
        ax3.fill_between(x_vals, stats.norm.pdf(x_vals),
                         where=(x_vals >= z_critico),  color="red", alpha=0.4)
    elif tipo_prueba == "Cola izquierda (<)":
        ax3.fill_between(x_vals, stats.norm.pdf(x_vals),
                         where=(x_vals <= z_critico), color="red", alpha=0.4, label="Zona de rechazo")
    else:
        ax3.fill_between(x_vals, stats.norm.pdf(x_vals),
                         where=(x_vals >= z_critico), color="red", alpha=0.4, label="Zona de rechazo")

    ax3.axvline(Z, color="blue", linestyle="--", linewidth=2, label=f"Z calculado = {Z:.3f}")
    ax3.set_title("Distribución normal con zona de rechazo")
    ax3.legend()
    st.pyplot(fig3)

    # ── MÓDULO 4: IA ──────────────────────────────────────────────
    st.header("4. Análisis con IA (Gemini)")
    api_key = st.text_input("Ingresa tu API Key de Gemini", type="password")

    if api_key and st.button("Consultar a Gemini"):
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"""Se realizó una prueba Z con los siguientes parámetros:
- Media muestral: {x_barra:.4f}
- Media hipotética (H₀): {mu0}
- Tamaño de muestra: {n}
- Desviación estándar poblacional: {sigma}
- Nivel de significancia: {alpha}
- Tipo de prueba: {tipo_prueba}
- Estadístico Z: {Z:.4f}
- p-value: {p_value:.4f}
- Decisión: {'Se rechaza H₀' if rechazar else 'No se rechaza H₀'}

¿Se rechaza H₀? Explica la decisión en términos simples y si los supuestos de la prueba son razonables."""
            respuesta = model.generate_content(prompt)
            st.write("**Respuesta de Gemini:**")
            st.info(respuesta.text)
            st.write("**Mi decisión vs IA:**")
            st.write(f"Yo decidí: {'Rechazar H₀' if rechazar else 'No rechazar H₀'}")
            st.write(f"La IA analizó los mismos datos y confirmó o complementó esta decisión.")
        except Exception as e:
            st.error(f"Error con Gemini: {e}")