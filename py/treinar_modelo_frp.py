import pandas as pd
import numpy as np
import os
import joblib
import warnings
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

warnings.filterwarnings('ignore')

DATA_DIR = r'C:\Users\vitor\OneDrive\Documentos\Github\previsao_incendios\dados'
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modelo_frp')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHUNK_SIZE = 100000

print("=" * 60)
print("TREINAMENTO - Regressão FRP (Fire Radiative Power)")
print("Sem amostras sintéticas, sem data leakage")
print("=" * 60)

print("\nProcessando dados chunk por chunk...")
arquivos = [f for f in os.listdir(DATA_DIR) if f.startswith('focos_br') and f.endswith('.csv')]
partial_aggs = []
chart_biomas_acc = None
chart_meses_acc = None
chart_estados_acc = None
municipio_acc = {}

t0 = time.time()

for arquivo in arquivos:
    caminho = os.path.join(DATA_DIR, arquivo)
    print(f"  >>> {arquivo}")
    for i, chunk in enumerate(pd.read_csv(caminho, sep=',', encoding='utf-8', chunksize=CHUNK_SIZE)):

        chunk['data'] = pd.to_datetime(chunk['data_pas'], errors='coerce')
        chunk['mes'] = chunk['data'].dt.month
        chunk['ano'] = chunk['data'].dt.year
        chunk['semana'] = chunk['data'].dt.isocalendar().week.astype(int)

        for col in ['numero_dias_sem_chuva', 'precipitacao', 'frp']:
            chunk[col] = chunk[col].where(chunk[col] >= 0, 0).fillna(0)

        if chart_biomas_acc is None:
            chart_biomas_acc = chunk['bioma'].value_counts()
            chart_meses_acc = chunk['mes'].value_counts()
            chart_estados_acc = chunk['estado'].value_counts()
        else:
            chart_biomas_acc = chart_biomas_acc.add(chunk['bioma'].value_counts(), fill_value=0)
            chart_meses_acc = chart_meses_acc.add(chunk['mes'].value_counts(), fill_value=0)
            chart_estados_acc = chart_estados_acc.add(chunk['estado'].value_counts(), fill_value=0)

        for (m, e, b), grp in chunk.groupby(['municipio', 'estado', 'bioma']):
            key = (m, e, b)
            lat_s, lon_s, c = municipio_acc.get(key, (0.0, 0.0, 0))
            municipio_acc[key] = (lat_s + grp['latitude'].sum(), lon_s + grp['longitude'].sum(), c + len(grp))

        chunk_agg = chunk.groupby(['semana', 'municipio', 'estado', 'ano', 'mes', 'bioma']).agg(
            frp_medio=('frp', 'mean'),
            frp_max=('frp', 'max'),
            precipitacao_total=('precipitacao', 'sum'),
            dias_sem_chuva_max=('numero_dias_sem_chuva', 'max'),
            latitude=('latitude', 'mean'),
            longitude=('longitude', 'mean'),
        ).reset_index()

        partial_aggs.append(chunk_agg)

        if (i + 1) % 20 == 0:
            print(f"    chunk {i+1:3d} | {len(chunk_agg):>5d} agg | {time.time()-t0:.0f}s")

print(f"\nAgregações parciais: {len(partial_aggs)} chunks")
df_agg = pd.concat(partial_aggs, ignore_index=True)
del partial_aggs

print("Segunda agregação global...")
df = df_agg.groupby(['semana', 'municipio', 'estado', 'ano', 'mes', 'bioma']).agg(
    frp_medio=('frp_medio', 'mean'),
    frp_max=('frp_max', 'max'),
    precipitacao_total=('precipitacao_total', 'sum'),
    dias_sem_chuva_max=('dias_sem_chuva_max', 'max'),
    latitude=('latitude', 'mean'),
    longitude=('longitude', 'mean'),
).reset_index()
del df_agg

print(f"Registros únicos: {len(df):,}")

df['semana_sin'] = np.sin(2 * np.pi * df['semana'] / 52)
df['semana_cos'] = np.cos(2 * np.pi * df['semana'] / 52)
df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)

print("Codificando bioma...")
encoders = {}
le = LabelEncoder()
df['bioma'] = le.fit_transform(df['bioma'].astype(str))
encoders['bioma'] = le
biomas_ordenados = sorted(le.classes_)

feature_cols = ['mes', 'mes_sin', 'mes_cos', 'semana_sin', 'semana_cos',
                'precipitacao_total', 'dias_sem_chuva_max',
                'bioma', 'latitude', 'longitude']

target_col = 'frp_medio'

print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
print(f"Target: {target_col}")

print("\nSeparando treino (2023-2024) e teste (2025)...")
df_train = df[df['ano'].isin([2023, 2024])].copy()
df_test = df[df['ano'] == 2025].copy()
print(f"  Treino: {len(df_train):,} registros")
print(f"  Teste:  {len(df_test):,} registros")

X_train = df_train[feature_cols].astype(float)
y_train = df_train[target_col].values
X_test = df_test[feature_cols].astype(float)
y_test = df_test[target_col].values

print(f"\nDistribuição do FRP no treino:")
print(f"  Média: {np.mean(y_train):.2f} | Mediana: {np.median(y_train):.2f}")
print(f"  P25: {np.percentile(y_train, 25):.2f} | P75: {np.percentile(y_train, 75):.2f}")
print(f"  Min: {np.min(y_train):.2f} | Max: {np.max(y_train):.2f}")

print("\nTreinando XGBoost Regressor...")
model = xgb.XGBRegressor(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.8, random_state=42,
    reg_lambda=1, reg_alpha=0.1, verbosity=0
)

t0 = time.time()
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print(f"  Treino concluído em {time.time()-t0:.0f}s")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n=== Métricas no teste (2025) ===")
print(f"  MAE:  {mae:.3f}")
print(f"  RMSE: {rmse:.3f}")
print(f"  R²:   {r2:.4f}")
print(f"  FRP médio real: {np.mean(y_test):.3f}")
print(f"  FRP médio pred: {np.mean(y_pred):.3f}")

p25 = np.percentile(y_train, 33)
p75 = np.percentile(y_train, 66)
print(f"\nLimiares de risco (baseado nos percentis do treino):")
print(f"  Baixo  (< P33): < {p25:.2f}")
print(f"  Médio  (P33-P66): {p25:.2f} - {p75:.2f}")
print(f"  Alto   (> P66): > {p75:.2f}")

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
for _, r in feature_importance.iterrows():
    print(f"  {r['feature']}: {r['importance']:.4f}")

print("\nSalvando artefatos...")

municipio_records = []
for (m, e, b), (lat_s, lon_s, c) in municipio_acc.items():
    municipio_records.append({
        'municipio': m, 'estado': e, 'bioma': b,
        'latitude': lat_s / c, 'longitude': lon_s / c
    })
municipio_info = pd.DataFrame(municipio_records)
del municipio_acc

print("Gerando previsões para o mapa...")
df_map = df.groupby('municipio', as_index=False).agg({
    'mes': 'last',
    'precipitacao_total': 'mean',
    'dias_sem_chuva_max': 'mean',
    'bioma': 'last',
    'latitude': 'mean',
    'longitude': 'mean',
})

semana_alvo = (pd.Timestamp.now().isocalendar().week + 1) % 52 or 52

df_map['semana_sin'] = np.sin(2 * np.pi * semana_alvo / 52)
df_map['semana_cos'] = np.cos(2 * np.pi * semana_alvo / 52)
df_map['mes_sin'] = np.sin(2 * np.pi * df_map['mes'] / 12)
df_map['mes_cos'] = np.cos(2 * np.pi * df_map['mes'] / 12)
# bioma ja esta codificado em df

X_map = df_map[feature_cols].astype(float)
df_map['frp_previsto'] = model.predict(X_map)

def categorizar_risco(frp):
    if frp < p25:
        return 'Baixo'
    elif frp < p75:
        return 'Médio'
    return 'Alto'

df_map['risco'] = df_map['frp_previsto'].apply(categorizar_risco)

map_final = municipio_info.merge(
    df_map[['municipio', 'frp_previsto', 'risco', 'semana_sin', 'semana_cos', 'precipitacao_total', 'dias_sem_chuva_max']],
    on='municipio', how='inner'
)
map_final = map_final.drop_duplicates(subset=['municipio'])
map_final.to_csv(os.path.join(OUTPUT_DIR, 'mapa_previsoes.csv'), index=False)
print(f"  Mapa salvo com {len(map_final)} municípios")

chart_biomas = chart_biomas_acc.sort_values(ascending=False)
chart_meses = chart_meses_acc.sort_index()
chart_estados = chart_estados_acc.sort_values(ascending=False)

artifactos = {
    'xgb_model.pkl': model,
    'label_encoders.pkl': encoders,
    'municipio_info.pkl': municipio_info,
    'biomas_list.pkl': biomas_ordenados,
    'feature_importance.pkl': feature_importance,
    'feature_cols.pkl': feature_cols,
    'chart_biomas.pkl': chart_biomas,
    'chart_meses.pkl': chart_meses,
    'chart_estados.pkl': chart_estados,
    'limiares_risco.pkl': {'baixo': float(p25), 'alto': float(p75)},
}

for nome, obj in artifactos.items():
    joblib.dump(obj, os.path.join(OUTPUT_DIR, nome))

print(f"\n{'=' * 60}")
print("TREINAMENTO CONCLUÍDO!")
print(f"{'=' * 60}")
print(f"  Registros processados: {len(df):,}")
print(f"  Treino (2023-24): {len(df_train):,}")
print(f"  Teste  (2025):    {len(df_test):,}")
print(f"  MAE: {mae:.3f} | R²: {r2:.4f}")
print(f"  Modelo salvo em: {OUTPUT_DIR}")
