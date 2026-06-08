import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

st.set_page_config(
    page_title="Previsão de Incêndios - Brasil",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODELO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modelo')

@st.cache_data
def carregar_artefatos():
    model = joblib.load(os.path.join(MODELO_DIR, 'xgb_model.pkl'))
    encoders = joblib.load(os.path.join(MODELO_DIR, 'label_encoders.pkl'))
    biomas = joblib.load(os.path.join(MODELO_DIR, 'biomas_list.pkl'))
    feature_importance = joblib.load(os.path.join(MODELO_DIR, 'feature_importance.pkl'))
    corr_matrix = joblib.load(os.path.join(MODELO_DIR, 'corr_matrix.pkl'))
    chart_biomas = joblib.load(os.path.join(MODELO_DIR, 'chart_biomas.pkl'))
    chart_meses = joblib.load(os.path.join(MODELO_DIR, 'chart_meses.pkl'))
    chart_estados = joblib.load(os.path.join(MODELO_DIR, 'chart_estados.pkl'))
    feature_cols = joblib.load(os.path.join(MODELO_DIR, 'feature_cols.pkl'))
    mapa = pd.read_csv(os.path.join(MODELO_DIR, 'mapa_previsoes.csv'))
    return model, encoders, biomas, feature_importance, corr_matrix, chart_biomas, chart_meses, chart_estados, feature_cols, mapa

model, encoders, biomas, feature_importance, corr_matrix, chart_biomas, chart_meses, chart_estados, feature_cols, df_mapa = carregar_artefatos()

meses_pt = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
            'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']

st.title("🔥 Previsão de Incêndios no Brasil")
st.markdown("Modelo XGBoost treinado com dados de satélite (INPE) para prever probabilidade de focos de incêndio.")

tab_mapa, tab_previsao, tab_graficos = st.tabs(["🗺️ Mapa de Calor", "📝 Previsão Manual", "📊 Gráficos"])

# ============================================================
# TAB 1: MAPA DE CALOR
# ============================================================
with tab_mapa:
    st.subheader(f"Probabilidade de Incêndio — Semana {df_mapa['semana'].iloc[0]} (próxima semana)")

    estados_list = sorted(df_mapa['estado'].unique())
    estado_filtro = st.multiselect("Filtrar por estado", estados_list, default=[])

    df_map_filtrado = df_mapa.copy()
    if estado_filtro:
        df_map_filtrado = df_map_filtrado[df_map_filtrado['estado'].isin(estado_filtro)]

    col1, col2, col3 = st.columns(3)
    col1.metric("Municípios", len(df_map_filtrado))
    col2.metric("Risco Alto (>75%)", f"{(df_map_filtrado['probabilidade'] > 0.75).sum()}")
    col3.metric("Risco Baixo (<25%)", f"{(df_map_filtrado['probabilidade'] < 0.25).sum()}")

    fig_mapa = px.scatter_map(
        df_map_filtrado,
        lat='latitude',
        lon='longitude',
        color='probabilidade',
        size='probabilidade',
        size_max=12,
        color_continuous_scale=px.colors.sequential.YlOrRd,
        range_color=[0, 1],
        hover_name='municipio',
        hover_data={'estado': True, 'probabilidade': ':.1%', 'latitude': False, 'longitude': False},
        map_style='carto-darkmatter',
        zoom=3,
        center={'lat': -14.2, 'lon': -51.9},
        title='Probabilidade de Incêndio por Município'
    )
    fig_mapa.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=30, b=0),
        coloraxis_colorbar=dict(title="Probabilidade", tickformat=".0%")
    )
    st.plotly_chart(fig_mapa, width='stretch')

# ============================================================
# TAB 2: PREVISÃO MANUAL
# ============================================================
with tab_previsao:
    st.subheader("Insira os parâmetros para prever a probabilidade de incêndio")

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

    if st.button("🔮 Prever Probabilidade", type="primary", use_container_width=True):
        semana_sin = np.sin(2 * np.pi * semana_input / 52)
        semana_cos = np.cos(2 * np.pi * semana_input / 52)
        bioma_encoded = encoders['bioma'].transform([bioma_input])[0]

        features = pd.DataFrame([[
            mes_input, precip_input, dias_sem_chuva,
            bioma_encoded, semana_sin, semana_cos
        ]], columns=feature_cols)

        prob = model.predict_proba(features.astype(float))[0][1]
        classe = "🔥 Alto risco" if prob >= 0.5 else "✅ Baixo risco"

        st.markdown("---")
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("Probabilidade", f"{prob:.1%}")
        col_res2.metric("Classificação", classe)
        col_res3.metric("Semana", semana_input)

        st.progress(float(prob))

        if prob >= 0.75:
            st.error("⚠️ Probabilidade alta de incêndio. Recomenda-se monitoramento intensivo.")
        elif prob >= 0.5:
            st.warning("⚠️ Probabilidade moderada de incêndio. Mantenha atenção.")
        else:
            st.success("✅ Baixa probabilidade de incêndio.")

# ============================================================
# TAB 3: GRÁFICOS
# ============================================================
with tab_graficos:
    st.subheader("📊 Análises Exploratórias dos Dados de Focos de Incêndio (INPE)")

    aba1, aba2, aba3, aba4, aba5 = st.tabs(
        ["Biomas", "Meses", "Estados", "Correlação", "Feature Importance"]
    )

    cores_neon = ['#8B0000', '#B22222', '#FF4500', '#FF6347', '#FFD700', '#FFFF00']

    with aba1:
        fig = px.bar(
            x=chart_biomas.index, y=chart_biomas.values,
            color=chart_biomas.index,
            color_discrete_sequence=cores_neon[:len(chart_biomas)],
            labels={'x': 'Bioma', 'y': 'Número de Focos'},
            title='Biomas com mais focos de incêndio'
        )
        fig.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig, width='stretch')

    with aba2:
        fig2 = px.bar(
            x=chart_meses.index.astype(str), y=chart_meses.values,
            color=chart_meses.index.astype(str),
            color_discrete_sequence=cores_neon * 2,
            labels={'x': 'Mês', 'y': 'Número de Focos'},
            title='Meses com mais focos de incêndio'
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
            title='Estados com mais focos de incêndio'
        )
        fig3.update_layout(showlegend=False, height=600, xaxis_tickangle=-90)
        st.plotly_chart(fig3, width='stretch')

    with aba4:
        var_names = {
            'incendio_total': 'Total Incêndios',
            'risco_fogo_medio': 'Risco Fogo Médio',
            'precipitacao_total': 'Precipitação Total',
            'dias_sem_chuva_max': 'Dias sem Chuva (máx)'
        }
        corr_renamed = corr_matrix.rename(index=var_names, columns=var_names)
        fig4 = px.imshow(
            corr_renamed.values,
            x=corr_renamed.columns, y=corr_renamed.index,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title='Mapa de Correlação',
            zmin=-1, zmax=1
        )
        fig4.update_layout(height=500)
        st.plotly_chart(fig4, width='stretch')

    with aba5:
        fig5 = px.bar(
            feature_importance, x='importance', y='feature',
            orientation='h',
            color='importance',
            color_continuous_scale='YlOrRd',
            title='Importância das Features (XGBoost)',
            labels={'importance': 'Importância', 'feature': 'Feature'}
        )
        fig5.update_layout(height=400)
        st.plotly_chart(fig5, width='stretch')

st.markdown("---")
st.caption("Dados: INPE (Programa Queimadas) • Modelo: XGBoost • Desenvolvido com Streamlit")
