import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import numpy as np

# ============================================
# CARGA DE DATOS
# ============================================
df = pd.read_csv("datos_procesados.csv")

# ============================================
# CONFIGURACIÓN DE PÁGINA
# ============================================
st.set_page_config(page_title="Videojuegos Dashboard", layout="wide")

# ============================================
# SIDEBAR – FILTROS
# ============================================
st.sidebar.title("🎮 Filtros")

plataformas = df["Plataforma"].unique()
filtro_plataforma = st.sidebar.multiselect(
    "Selecciona una o varias plataformas:",
    plataformas,
    default=plataformas[:3]
)

generos = df["Genero"].unique()
filtro_genero = st.sidebar.multiselect(
    "Selecciona géneros:",
    generos,
    default=generos[:4]
)

df_filtrado = df[
    (df["Plataforma"].isin(filtro_plataforma)) &
    (df["Genero"].isin(filtro_genero))
]

# ============================================
# HEADER
# ============================================
st.title("🎮 Dashboard Interactivo de Ventas de Videojuegos")

col1, col2, col3 = st.columns(3)
col1.metric("Juegos filtrados", len(df_filtrado))
col2.metric("Ventas globales (M)", round(df_filtrado["Ventas Global"].sum(), 2))
col3.metric("Año más reciente", int(df_filtrado["Año"].max()))

# ============================================
# GRÁFICO 1 – Ventas por género
# ============================================
st.subheader("📊 Ventas globales por género")

ventas_genero = df_filtrado.groupby("Genero")["Ventas Global"].sum().reset_index()

fig1 = px.bar(
    ventas_genero,
    x="Genero",
    y="Ventas Global",
    title="Ventas por Género",
    text="Ventas Global"
)
st.plotly_chart(fig1, use_container_width=True)

# ============================================
# GRÁFICO 2 – Ventas por plataforma
# ============================================
st.subheader("🎮 Ventas por Plataforma")

ventas_plataforma = df_filtrado.groupby("Plataforma")["Ventas Global"].sum().reset_index()

fig2 = px.pie(
    ventas_plataforma,
    names="Plataforma",
    values="Ventas Global",
    hole=0.4,
    title="Distribución de Ventas"
)
st.plotly_chart(fig2, use_container_width=True)

# ============================================
# GRÁFICO 3 – Evolución por año
# ============================================
st.subheader("📈 Evolución de ventas por año")

ventas_tiempo = df_filtrado.groupby("Año")["Ventas Global"].sum().reset_index()

fig3 = px.line(
    ventas_tiempo,
    x="Año",
    y="Ventas Global",
    markers=True,
    title="Tendencia de Ventas"
)

st.plotly_chart(fig3, use_container_width=True)

# ============================================
# TABLA – Datos filtrados
# ============================================
st.subheader("📄 Datos filtrados")
st.dataframe(df_filtrado)

# ============================================
# GRÁFICO 4 – Top 10 juegos más vendidos
# ============================================
st.subheader("🏆 Top 10 juegos más vendidos")

top10 = df_filtrado.sort_values("Ventas Global", ascending=False).head(10)

fig4 = px.bar(
    top10,
    x="Ventas Global",
    y="Nombre",
    orientation="h",
    text="Ventas Global",
    title="Top 10 Juegos Más Vendidos"
)
st.plotly_chart(fig4, use_container_width=True)

# ============================================
# GRÁFICO 5 – Mapa de calor de correlación
# ============================================
st.subheader("🔥 Mapa de correlación entre variables numéricas")

corr = df_filtrado.select_dtypes(include="number").corr()

fig5 = px.imshow(
    corr,
    text_auto=True,
    title="Correlación entre variables"
)
st.plotly_chart(fig5, use_container_width=True)

# ============================================
# GRÁFICO 6 – Dispersión Año vs Ventas Globales
# ============================================
st.subheader("📌 Relación entre Año y Ventas Globales")

fig6 = px.scatter(
    df_filtrado,
    x="Año",
    y="Ventas Global",
    color="Genero",
    size="Ventas Global",
    title="Año vs Ventas Globales"
)
st.plotly_chart(fig6, use_container_width=True)

# ============================================
# GRÁFICO 7 – Participación por Editorial
# ============================================
st.subheader("🏢 Participación por Editorial")

ventas_editorial = df_filtrado.groupby("Editorial")["Ventas Global"].sum().reset_index()

fig7 = px.pie(
    ventas_editorial,
    names="Editorial",
    values="Ventas Global",
    title="Participación por Editorial en Ventas"
)
st.plotly_chart(fig7, use_container_width=True)

# ============================================
# RANKING – Top 20 juegos más vendidos
# ============================================
st.subheader("🏆 Top 20 Juegos Más Vendidos")

top20 = df_filtrado.sort_values(by="Ventas Global", ascending=False).head(20)

fig_top20 = px.bar(
    top20,
    x="Nombre",
    y="Ventas Global",
    text="Ventas Global",
    title="Top 20 Juegos Más Vendidos",
)
fig_top20.update_layout(xaxis_tickangle=-45)

st.plotly_chart(fig_top20, use_container_width=True)

# ============================================
# SECCIÓN NUEVA – PREDICTOR ML (CLASIFICACIÓN)
# ============================================
# ============================================
# SECCIÓN NUEVA – PREDICTOR ML (CLASIFICACIÓN)
# ============================================

st.markdown("---")
st.subheader("🤖 Predictor de Éxito con Machine Learning")

st.info("""
### 📌 Sobre este Predictor

**¿Qué hace bien?**
- ✅ Identifica patrones históricos (1980-2016)
- ✅ Funciona bien para años dentro del rango de entrenamiento
- ✅ Útil para entender qué funcionó en el pasado

**Limitaciones:**
- ⚠️ **Años 2017+**: El modelo no fue entrenado con datos recientes, por lo que las predicciones tienen alta incertidumbre
- ⚠️ **Mercado cambió**: No considera juegos digitales, F2P, streaming, Game Pass
- ⚠️ **Desbalance**: 76% de juegos históricamente fracasaron, el modelo tiende a ser pesimista

**Mejor uso:** Análisis retrospectivo y comprensión de factores históricos de éxito.
""")

import gzip

@st.cache_resource
def cargar_modelo():
    try:
        with gzip.open("modelo_entrenado.pkl.gz", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        return None


modelo_data = cargar_modelo()

if modelo_data is None:
    st.error("⚠️ Modelo no disponible. Ejecuta 'analisis_completo.py' primero para generar 'modelo_entrenado.pkl'.")
else:
    clf = modelo_data['clasificador']
    le_gen = modelo_data['le_genero']
    le_plat = modelo_data['le_plataforma']
    le_edit = modelo_data['le_editorial']
    le_cat = modelo_data['le_categoria']

    # Métricas del modelo
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Precisión (Accuracy)", f"{modelo_data['metricas_clasificacion']['accuracy']*100:.1f}%")
    with col2:
        st.metric("Cross-Validation", f"{modelo_data['metricas_clasificacion']['cross_val']*100:.1f}%")
    with col3:
        factor_top = modelo_data['importancias'][0]['Variable']
        st.metric("Factor Clave", factor_top)

    st.markdown("### 🎯 Predice el Éxito de un Nuevo Juego")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        gen_pred = st.selectbox("🎮 Género", sorted(le_gen.classes_.tolist()))
    with col2:
        plat_pred = st.selectbox("🕹️ Plataforma", sorted(le_plat.classes_.tolist()))
    with col3:
        edit_pred = st.selectbox("🏢 Editorial", sorted(le_edit.classes_.tolist()))
    with col4:
        año_pred = st.number_input("📅 Año", min_value=1980, max_value=2025, value=2016)

    if st.button("🚀 PREDECIR CATEGORÍA DE ÉXITO", type="primary", use_container_width=True):
        try:
            gen_enc = le_gen.transform([gen_pred])[0]
            plat_enc = le_plat.transform([plat_pred])[0]
            edit_enc = le_edit.transform([edit_pred])[0]

            features = np.array([[gen_enc, plat_enc, edit_enc, año_pred]])
            pred_enc = clf.predict(features)[0]
            pred_proba = clf.predict_proba(features)[0]
            categoria_pred = le_cat.inverse_transform([pred_enc])[0]

            colors = {'Fracaso': '#e74c3c', 'Moderado': '#f39c12', 'Éxito': '#3498db', 'Blockbuster': '#2ecc71'}
            
            # Mostrar las probabilidades ordenadas
            prob_ordenadas = sorted(zip(modelo_data["categorias"], pred_proba), key=lambda x: x[1], reverse=True)

            st.warning("⚠️ **Nota importante:** El modelo tiene baja confianza en predicciones para años recientes (2017+). Las probabilidades están muy equilibradas, lo que indica incertidumbre del modelo.")

            st.markdown("### 📊 Distribución de Probabilidades")

            for i, (cat, prob) in enumerate(prob_ordenadas):
                color_cat = colors.get(cat, '#95a5a6')
                porcentaje = prob * 100
                
                # Icono según ranking
                icono = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📊"
                
                st.markdown(f"""
                <div style='background: {color_cat}; padding: 15px; border-radius: 10px; margin: 10px 0; opacity: {1 - (i*0.2)};'>
                    <span style='font-size: 1.5rem;'>{icono}</span>
                    <span style='color: white; font-size: 1.3rem; font-weight: bold;'> {cat}</span>
                    <span style='float: right; color: white; font-size: 1.3rem; font-weight: bold;'>{porcentaje:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

            # Añadir interpretación inteligente
            diferencia = prob_ordenadas[0][1] - prob_ordenadas[1][1]

            if diferencia < 0.05:  # Menos de 5% de diferencia
                st.error(f"""
🚨 **Confianza MUY BAJA**: La diferencia entre las dos categorías más probables es solo {diferencia*100:.1f}%. 

**Razón:** El modelo fue entrenado con datos hasta 2016. Para años recientes (2017+), tiene dificultad 
para predecir porque el mercado cambió radicalmente (juegos digitales, F2P, Game Pass, etc.).

**Recomendación:** Esta predicción debe tomarse solo como referencia histórica, no como pronóstico confiable.
                """)
            elif diferencia < 0.15:  # Menos de 15%
                st.warning(f"""
⚠️ **Confianza MODERADA**: Hay {diferencia*100:.1f}% de diferencia entre las dos principales categorías.
El modelo tiene cierta incertidumbre en esta predicción.
                """)
            else:
                st.success(f"""
✅ **CONFIANZA ALTA**: El modelo tiene {diferencia*100:.1f}% de diferencia clara entre categorías.
Esta combinación tiene patrones históricos definidos.
                """)

            # Gráfico de barras de probabilidades
            prob_df = pd.DataFrame({
                "Categoría": modelo_data["categorias"],
                "Probabilidad": pred_proba
            }).sort_values("Probabilidad", ascending=False)

            fig_prob = px.bar(
                prob_df,
                x="Probabilidad",
                y="Categoría",
                orientation="h",
                color="Probabilidad",
                color_continuous_scale="RdYlGn"
            )
            fig_prob.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_prob, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error en la predicción: {e}")