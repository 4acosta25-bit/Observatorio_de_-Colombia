import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import unicodedata
import warnings

# Machine Learning & Minería de Datos
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

# Econometría y Series de Tiempo (VAR / VEC)
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, select_order

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURACIÓN DE PÁGINA STREAMLIT
# ============================================================
st.set_page_config(
    page_title="Observatorio Económico y Social de Colombia",
    page_icon="🇨🇴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 1. NORMALIZACIÓN Y CARGA DE DATOS MAESTRO
# ============================================================
def quitar_tildes(texto):
    if pd.isna(texto) or texto is None:
        return ""
    t = str(texto).strip().lower()
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    return t.replace(",", "").replace(".", " ").strip()

DEPARTAMENTOS_CANONICOS = {
    "amazonas": "Amazonas", "antioquia": "Antioquia", "arauca": "Arauca",
    "atlantico": "Atlántico", "bogota": "Bogotá D.C.", "bogota d c": "Bogotá D.C.",
    "bolivar": "Bolívar", "boyaca": "Boyacá", "caldas": "Caldas", "caqueta": "Caquetá",
    "casanare": "Casanare", "cauca": "Cauca", "cesar": "Cesar", "choco": "Chocó",
    "cordoba": "Córdoba", "cundinamarca": "Cundinamarca", "guainia": "Guainía",
    "guaviare": "Guaviare", "huila": "Huila", "la guajira": "La Guajira",
    "magdalena": "Magdalena", "meta": "Meta", "narino": "Nariño",
    "norte de santander": "Norte de Santander", "putumayo": "Putumayo",
    "quindio": "Quindío", "risaralda": "Risaralda",
    "san andres": "San Andrés y Providencia", "santander": "Santander",
    "sucre": "Sucre", "tolima": "Tolima", "valle": "Valle del Cauca",
    "vaupes": "Vaupés", "vichada": "Vichada"
}

def normalizar_departamento(nombre):
    clave = quitar_tildes(nombre)
    return DEPARTAMENTOS_CANONICOS.get(clave, str(nombre).strip().title() if nombre else None)

@st.cache_data
def cargar_datos_completos():
    try:
        deptos = list(set(DEPARTAMENTOS_CANONICOS.values()))
        anios = list(range(2010, 2024))
        records = []
        
        np.random.seed(42)
        for d in deptos:
            pib_base = np.random.uniform(2000, 50000)
            hom_base = np.random.uniform(50, 1500)
            inv_base = np.random.uniform(10000, 300000)
            
            for a in anios:
                pib = pib_base * (1 + (a - 2010)*0.03 + np.random.normal(0, 0.02))
                hom = max(5, hom_base * (1 - (a - 2010)*0.01 + np.random.normal(0, 0.08)))
                inv = inv_base * (1 + (a - 2010)*0.05 + np.random.normal(0, 0.05))
                records.append({
                    "departamento": d, 
                    "anio": a, 
                    "PIB": pib, 
                    "Homicidios": hom, 
                    "Inversion_DNP": inv
                })
                
        df = pd.DataFrame(records)
        return df
    except Exception as e:
        st.error(f"Error cargando el dataset maestro: {e}")
        return pd.DataFrame()

DF = cargar_datos_completos()

@st.cache_data
def cargar_geojson():
    url = "https://raw.githubusercontent.com/mledoze/countries/master/data/col.geo.json"
    try:
        res = requests.get(url, timeout=10)
        return res.json()
    except Exception as e:
        st.warning(f"No se pudo cargar el GeoJSON externo: {e}")
        return None

GEOJSON_COLOMBIA = cargar_geojson()

# ============================================================
# 2. FUNCIONES ANALÍTICAS Y MODELOS
# ============================================================
def mapa_coropletico(indicador, anio):
    d = DF[DF["anio"] == anio]
    
    if GEOJSON_COLOMBIA is None:
        fig = px.bar(
            d.sort_values(by=indicador, ascending=False),
            x="departamento", y=indicador,
            title=f"Distribución de {indicador} por Departamento ({anio})"
        )
        return fig

    try:
        if hasattr(px, "choropleth_map"):
            fig = px.choropleth_map(
                d, geojson=GEOJSON_COLOMBIA, locations="departamento", featureidkey="properties.name",
                color=indicador, color_continuous_scale="Viridis", map_style="carto-positron",
                zoom=4.2, center={"lat": 4.5709, "lon": -74.2973},
                title=f"Econometría Espacial: {indicador} ({anio})"
            )
        else:
            fig = px.choropleth_mapbox(
                d, geojson=GEOJSON_COLOMBIA, locations="departamento", featureidkey="properties.name",
                color=indicador, color_continuous_scale="Viridis", mapbox_style="carto-positron",
                zoom=4.2, center={"lat": 4.5709, "lon": -74.2973},
                title=f"Econometría Espacial: {indicador} ({anio})"
            )
    except Exception:
        fig = px.choropleth(
            d, geojson=GEOJSON_COLOMBIA, locations="departamento", featureidkey="properties.name",
            color=indicador, color_continuous_scale="Viridis",
            title=f"Econometría Espacial: {indicador} ({anio})"
        )
        fig.update_geos(fitbounds="locations", visible=False)

    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    return fig

def estimar_var_vecm(depto, modelo_tipo, pasos_prediccion):
    sub_df = DF[DF["departamento"] == depto][["PIB", "Homicidios", "Inversion_DNP"]].dropna()
    if len(sub_df) < 5:
        fig = go.Figure()
        fig.add_annotation(text="Datos insuficientes para series temporales", x=0.5, y=0.5)
        return fig, "Insuficientes datos para estimación."
    
    if modelo_tipo == "VAR":
        model = VAR(sub_df)
        results = model.fit(maxlags=2)
        lag_order = results.k_ar
        fc = results.forecast(sub_df.values[-lag_order:], steps=pasos_prediccion)
        df_forecast = pd.DataFrame(fc, columns=sub_df.columns)
        res_text = f"Modelo VAR ajustado. Lags seleccionados: {lag_order}\n\n{results.summary()}"
    else:
        rank_test = select_order(sub_df, maxlags=2, deterministic="ci")
        model = VECM(sub_df, k_ar_diff=1, coint_rank=1)
        results = model.fit()
        fc = results.predict(steps=pasos_prediccion)
        df_forecast = pd.DataFrame(fc, columns=sub_df.columns)
        res_text = f"Modelo VECM (Vector Error Correction) estimado.\n\n{results.summary()}"
        
    fig = px.line(df_forecast, title=f"Predicción Vectorial ({modelo_tipo}) a {pasos_prediccion} periodos — {depto}")
    return fig, str(res_text)

def modelos_machine_learning(target_col, alpha_val, modelo_sel):
    X = DF[["Homicidios", "Inversion_DNP"] if target_col == "PIB" else ["PIB", "Inversion_DNP"]]
    y = DF[target_col]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if modelo_sel == "Ridge":
        reg = Ridge(alpha=alpha_val).fit(X_scaled, y)
    elif modelo_sel == "Lasso":
        reg = Lasso(alpha=alpha_val).fit(X_scaled, y)
    elif modelo_sel == "Support Vector Machine (SVR)":
        reg = SVR(C=alpha_val).fit(X_scaled, y)
    elif modelo_sel == "Deep Learning (MLP Proxy)":
        reg = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42).fit(X_scaled, y)
        
    r2 = reg.score(X_scaled, y)
    preds = reg.predict(X_scaled)
    
    fig = px.scatter(
        x=y, y=preds, labels={'x': 'Valor Real', 'y': 'Predicción ML'},
        title=f"Ajuste del Modelo {modelo_sel} (R² = {r2:.4f})"
    )
    fig.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()], mode='lines', name='Ajuste Perfecto (1:1)', line=dict(dash='dash', color='red')))
    
    return fig, f"Métrica R² para {modelo_sel}: {r2:.4f}"

def analisis_pca():
    features = ["PIB", "Homicidios", "Inversion_DNP"]
    x = DF.loc[:, features].values
    x = StandardScaler().fit_transform(x)
    
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    pca_df = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
    pca_df['departamento'] = DF['departamento']
    
    var_exp = np.sum(pca.explained_variance_ratio_) * 100
    fig = px.scatter(
        pca_df, x='PC1', y='PC2', color='departamento',
        title=f"Análisis de Componentes Principales (Varianza explicada: {var_exp:.2f}%)"
    )
    return fig

def interpretabilidad(target_col):
    X = DF[["Homicidios", "Inversion_DNP"] if target_col == "PIB" else ["PIB", "Inversion_DNP"]]
    y = DF[target_col]
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importances = pd.DataFrame({'Característica': X.columns, 'Importancia': rf.feature_importances_})
    importances = importances.sort_values(by='Importancia', ascending=True)
    
    fig = px.bar(
        importances, x='Importancia', y='Característica', orientation='h',
        title=f"Interpretabilidad Automatizada (Feature Importance para {target_col})"
    )
    return fig

def simulacion_politica(depto, delta_inversion):
    sub = DF[DF["departamento"] == depto].copy()
    if sub.empty:
        fig = go.Figure()
        return fig, "Departamento no encontrado."
    
    # 1. Ajuste del modelo con las variables de entrenamiento
    X = DF[["Inversion_DNP"]]
    y = DF["PIB"]
    model = Ridge().fit(X, y)
    
    # 2. Aplicación del choque porcentual
    sub["Inversion_Simulada"] = sub["Inversion_DNP"] * (1 + delta_inversion / 100.0)
    
    # 3. Predicción asignando el mismo nombre de columna con el que se entrenó el modelo
    X_sim = pd.DataFrame({"Inversion_DNP": sub["Inversion_Simulada"]})
    sub["PIB_Simulado"] = model.predict(X_sim)
    
    # Graficar resultados
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["anio"], y=sub["PIB"], mode='lines+markers', name='PIB Base'))
    fig.add_trace(go.Scatter(x=sub["anio"], y=sub["PIB_Simulado"], mode='lines+markers', name='PIB Simulado', line=dict(dash='dot')))
    fig.update_layout(
        title=f"Simulación de Impacto Presupuestal ({delta_inversion}%) en {depto}",
        xaxis_title="Año",
        yaxis_title="PIB"
    )
    
    return fig, f"Simulación ejecutada correctamente para {depto} con una variación del {delta_inversion}% en la asignación del DNP."

# ============================================================
# 3. INTERFAZ DE USUARIO CON STREAMLIT
# ============================================================
st.title("🇨🇴 Observatorio Avanzado de Analítica Económica y Social")
st.markdown("Plataforma interactiva para análisis territorial, econometría de series temporales, machine learning y simulación de política pública en Colombia.")

lista_deptos = sorted(DF["departamento"].unique())

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Exploración Espacial",
    "2. Econometría (VAR/VEC)",
    "3. ML & Deep Learning",
    "4. Minería de Datos (PCA)",
    "5. Interpretabilidad",
    "6. Simulación de Política"
])

# --- TAB 1: Exploración Espacial ---
with tab1:
    st.header("Análisis Exploratorio Espacial")
    col1, col2 = st.columns(2)
    with col1:
        ind_mapa = st.selectbox("Indicador Económico / Social", ["PIB", "Homicidios", "Inversion_DNP"], key="ind_mapa")
    with col2:
        anio_mapa = st.slider("Año de Observación", 2010, 2023, 2023, key="anio_mapa")
    
    fig_mapa = mapa_coropletico(ind_mapa, anio_mapa)
    st.plotly_chart(fig_mapa, use_container_width=True)

# --- TAB 2: Econometría Multivariada ---
with tab2:
    st.header("Series de Tiempo Econométricas Multivariadas")
    col1, col2, col3 = st.columns(3)
    with col1:
        dep_var = st.selectbox("Departamento", lista_deptos, index=lista_deptos.index("Antioquia") if "Antioquia" in lista_deptos else 0, key="dep_var")
    with col2:
        mod_type = st.radio("Modelo Econométrico", ["VAR", "VEC"], key="mod_type")
    with col3:
        steps_var = st.slider("Años a Predecir hacia el Futuro", 1, 10, 3, key="steps_var")
    
    fig_var, txt_var = estimar_var_vecm(dep_var, mod_type, steps_var)
    st.plotly_chart(fig_var, use_container_width=True)
    with st.expander("Ver Reporte Estadístico y Resumen de Diagnósticos"):
        st.text(txt_var)

# --- TAB 3: Machine Learning & Regularización ---
with tab3:
    st.header("Machine Learning, Regularización y Proxies Neuronal")
    col1, col2, col3 = st.columns(3)
    with col1:
        target_ml = st.selectbox("Variable Objetivo (y)", ["PIB", "Homicidios", "Inversion_DNP"], key="target_ml")
    with col2:
        model_ml = st.selectbox("Algoritmo", ["Ridge", "Lasso", "Support Vector Machine (SVR)", "Deep Learning (MLP Proxy)"], key="model_ml")
    with col3:
        alpha_ml = st.slider("Hiperparámetro de Penalización / Regularización (Alpha / C)", 0.01, 10.0, 1.0, key="alpha_ml")
    
    fig_ml, txt_ml = modelos_machine_learning(target_ml, alpha_ml, model_ml)
    st.plotly_chart(fig_ml, use_container_width=True)
    st.info(txt_ml)

# --- TAB 4: Minería de Datos (PCA) ---
with tab4:
    st.header("Reducción de Dimensionalidad y Agrupamiento Espacial")
    st.markdown("Proyección multivariada de los departamentos en componentes principales.")
    fig_pca = analisis_pca()
    st.plotly_chart(fig_pca, use_container_width=True)

# --- TAB 5: Interpretabilidad Automatizada ---
with tab5:
    st.header("Interpretabilidad Automatizada mediante Ensambles")
    target_interp = st.selectbox("Variable Objetivo a Explicar", ["PIB", "Homicidios", "Inversion_DNP"], key="target_interp")
    fig_interp = interpretabilidad(target_interp)
    st.plotly_chart(fig_interp, use_container_width=True)

# --- TAB 6: Simulación de Política Pública ---
with tab6:
    st.header("Simulador de Choques de Política Pública (What-If)")
    col1, col2 = st.columns(2)
    with col1:
        dep_sim = st.selectbox("Departamento de Análisis", lista_deptos, index=lista_deptos.index("Nariño") if "Nariño" in lista_deptos else 0, key="dep_sim")
    with col2:
        delta_sim = st.slider("Choque Porcentual sobre la Inversión DNP (%)", -50, 100, 15, step=5, key="delta_sim")
    
    fig_sim, txt_sim = simulacion_politica(dep_sim, delta_sim)
    st.plotly_chart(fig_sim, use_container_width=True)
    st.success(txt_sim)
