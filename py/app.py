import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

st.set_page_config(
    page_title="Previsão de Incêndios - Brasil (FRP)",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODELO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modelo_frp')

@st.cache_data
def carregar_artefatos():
    model = joblib.load(os.path.join(MODELO_DIR, 'xgb_model.pkl'))
    encoders = joblib.load(os.path.join(MODELO_DIR, 'label_encoders.pkl'))
    biomas = joblib.load(os.path.join(MODELO_DIR, 'biomas_list.pkl'))
    feature_importance = joblib.load(os.path.join(MODELO_DIR, 'feature_importance.pkl'))
    feature_cols = joblib.load(os.path.join(MODELO_DIR, 'feature_cols.pkl'))
    chart_biomas = joblib.load(os.path.join(MODELO_DIR, 'chart_biomas.pkl'))
    chart_meses = joblib.load(os.path.join(MODELO_DIR, 'chart_meses.pkl'))
    chart_estados = joblib.load(os.path.join(MODELO_DIR, 'chart_estados.pkl'))
    limiares = joblib.load(os.path.join(MODELO_DIR, 'limiares_risco.pkl'))
    mapa = pd.read_csv(os.path.join(MODELO_DIR, 'mapa_previsoes.csv'))
    return model, encoders, biomas, feature_importance, feature_cols, chart_biomas, chart_meses, chart_estados, limiares, mapa

model, encoders, biomas, feature_importance, feature_cols, chart_biomas, chart_meses, chart_estados, limiares, df_mapa = carregar_artefatos()

meses_pt = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
            'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']

def categorizar_risco(frp):
    if frp < limiares['baixo']:
        return 'Baixo'
    elif frp < limiares['alto']:
        return 'Médio'
    return 'Alto'

st.title("🔥 Previsão de Intensidade de Incêndios (FRP)")
st.markdown("Modelo XGBoost Regressor treinado com dados de satélite INPE. Prediz o **FRP (Fire Radiative Power)** médio esperado.")

tab_mapa, tab_previsao, tab_graficos = st.tabs(["🗺️ Mapa de Risco", "📝 Previsão Manual", "📊 Análises"])

with tab_mapa:
    st.subheader(f"Intensidade de Fogo Prevista (FRP) — Semana atual")

    municipios_ordenados = sorted(df_mapa['municipio'].unique())
    municipio_filtro = st.selectbox("Filtrar por município", [""] + municipios_ordenados)

    df_map_filtrado = df_mapa.copy()
    if municipio_filtro:
        df_map_filtrado = df_map_filtrado[df_map_filtrado['municipio'] == municipio_filtro]

    col1, col2, col3 = st.columns(3)
    col1.metric("Municípios", len(df_map_filtrado))
    col2.metric("Risco Alto", f"{(df_map_filtrado['risco'] == 'Alto').sum()}")
    col3.metric("Risco Baixo", f"{(df_map_filtrado['risco'] == 'Baixo').sum()}")

    fig_mapa = px.scatter_map(
        df_map_filtrado,
        lat='latitude',
        lon='longitude',
        color='frp_previsto',
        size='frp_previsto',
        size_max=14,
        color_continuous_scale='YlOrRd',
        hover_name='municipio',
        hover_data={
            'estado': True,
            'frp_previsto': ':.1f',
            'risco': True,
            'latitude': False,
            'longitude': False
        },
        map_style='carto-darkmatter',
        zoom=3,
        center={'lat': -14.2, 'lon': -51.9},
        title='Intensidade de Fogo Prevista (FRP) por Município'
    )
    fig_mapa.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=30, b=0),
        coloraxis_colorbar=dict(title="FRP previsto")
    )
    st.plotly_chart(fig_mapa, width='stretch')

with tab_previsao:
    st.subheader("Insira os parâmetros para prever a intensidade do fogo")

    col_esq, col_dir = st.columns([1, 1])

    with col_esq:
        semana_input = st.slider("Semana do ano", 1, 52, int(datetime.now().isocalendar().week))
        mes_input = st.selectbox("Mês", options=range(1, 13),
                                 format_func=lambda x: meses_pt[x - 1],
                                 index=datetime.now().month - 1)
        bioma_input = st.selectbox("Bioma", biomas)

    with col_dir:
        precip_input = st.number_input("Precipitação total (mm)", 0.0, 300.0, 30.0, step=1.0)
        dias_sem_chuva = st.number_input("Dias sem chuva (máx)", 0, 365, 5, step=1)
        municipio_filtro = st.selectbox("Filtrar por município", [""] + municipios_ordenados)

    if st.button("🔮 Prever Intensidade", type="primary", width='stretch'):
        semana_sin = np.sin(2 * np.pi * semana_input / 52)
        semana_cos = np.cos(2 * np.pi * semana_input / 52)
        mes_sin = np.sin(2 * np.pi * mes_input / 12)
        mes_cos = np.cos(2 * np.pi * mes_input / 12)
        bioma_encoded = encoders['bioma'].transform([bioma_input])[0]

        features = pd.DataFrame([[
            mes_input, mes_sin, mes_cos,
            semana_sin, semana_cos,
            precip_input, dias_sem_chuva,
            bioma_encoded
        ]], columns=feature_cols)

        frp_pred = model.predict(features.astype(float))[0]
        risco = categorizar_risco(frp_pred)

        st.markdown("---")
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("FRP Previsto", f"{frp_pred:.2f}")
        col_res2.metric("Classificação", risco)
        col_res3.metric("Semana", semana_input)

        pct = min(frp_pred / limiares['alto'], 1.0)
        st.progress(float(pct))

        if risco == 'Alto':
            st.error("⚠️ Risco ALTO de incêndio intenso. Monitoramento prioritário.")
        elif risco == 'Médio':
            st.warning("⚠️ Risco MÉDIO. Mantenha atenção.")
        else:
            st.success("✅ Risco BAIXO de incêndio intenso.")

with tab_graficos:
    st.subheader("📊 Análise Exploratória dos Focos de Incêndio (INPE)")
    st.caption("Dados reais de satélite — todas as ocorrências de 2023 a 2025")

    aba1, aba2, aba3, aba4 = st.tabs(
        ["Biomas", "Meses", "Estados", "Feature Importance"]
    )

    cores_neon = ['#8B0000', '#B22222', '#FF4500', '#FF6347', '#FFD700', '#FFFF00']

    with aba1:
        fig = px.bar(
            x=chart_biomas.index, y=chart_biomas.values,
            color=chart_biomas.index,
            color_discrete_sequence=cores_neon[:len(chart_biomas)],
            labels={'x': 'Bioma', 'y': 'Número de Focos'},
            title='Focos de incêndio por Bioma'
        )
        fig.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig, width='stretch')

    with aba2:
        fig2 = px.bar(
            x=chart_meses.index.astype(str), y=chart_meses.values,
            color=chart_meses.index.astype(str),
            color_discrete_sequence=cores_neon * 2,
            labels={'x': 'Mês', 'y': 'Número de Focos'},
            title='Focos de incêndio por Mês'
        )
        fig2.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig2, width='stretch')

    with aba3:
        estados_df = chart_estados.reset_index()
        estados_df.columns = ['estado', 'focos']
        fig3 = px.bar(
            estados_df, x='estado', y='focos',
            color='estado',
            color_discrete_sequence=px.colors.sequential.YlOrRd_r,
            title='Focos de incêndio por Estado'
        )
        fig3.update_layout(showlegend=False, height=600, xaxis_tickangle=-90)
        st.plotly_chart(fig3, width='stretch')

    with aba4:
        fig5 = px.bar(
            feature_importance, x='importance', y='feature',
            orientation='h',
            color='importance',
            color_continuous_scale='YlOrRd',
            title='Features que mais influenciam o FRP (XGBoost)',
            labels={'importance': 'Importância', 'feature': 'Feature'}
        )
        fig5.update_layout(height=500)
        st.plotly_chart(fig5, width='stretch')

st.markdown("---")
st.caption(f"Dados: INPE (Programa Queimadas) • Modelo: XGBoost Regressor (FRP) • Limiares: < {limiares['baixo']:.1f} Baixo / {limiares['baixo']:.1f}-{limiares['alto']:.1f} Médio / > {limiares['alto']:.1f} Alto")
