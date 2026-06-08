import pandas as pd
import numpy as np
import os
import joblib
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, log_loss
import xgboost as xgb

warnings.filterwarnings('ignore')

DATA_DIR = r'C:\Users\vitor\OneDrive\Documentos\Github\previsao_incendios\dados'
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modelo')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Carregando dados (amostra de 1% por chunk)...")
arquivos = [f for f in os.listdir(DATA_DIR) if f.startswith('focos_br') and f.endswith('.csv')]
dfs = []
for arquivo in arquivos:
    caminho = os.path.join(DATA_DIR, arquivo)
    for chunk in pd.read_csv(caminho, sep=',', encoding='utf-8', chunksize=100000):
        dfs.append(chunk.sample(frac=0.01, random_state=42))

df = pd.concat(dfs, ignore_index=True)
print(f"Total de amostras: {len(df)}")

print("Processando dados...")
df['data'] = pd.to_datetime(df['data_pas'], errors='coerce')
df['mes'] = df['data'].dt.month
df['ano'] = df['data'].dt.year
df['semana'] = df['data'].dt.isocalendar().week.astype(int)
df['dia'] = df['data'].dt.day

df['id_area_industrial'] = df['id_area_industrial'].apply(lambda x: 1 if x > 0 else 0)
df['numero_dias_sem_chuva'] = df['numero_dias_sem_chuva'].where(df['numero_dias_sem_chuva'] >= 0)
df['risco_fogo'] = df['risco_fogo'].where(df['risco_fogo'] >= 0)
df['numero_dias_sem_chuva'] = df['numero_dias_sem_chuva'].fillna(df['numero_dias_sem_chuva'].median())
df['risco_fogo'] = df['risco_fogo'].fillna(df['risco_fogo'].median())
df = df.rename(columns={'id_area_industrial': 'area_industrial'})
df = df.drop(columns=['data_pas'])

# Save chart data (from raw sample)
chart_biomas = df['bioma'].value_counts()
chart_meses = df['mes'].value_counts().sort_index()
chart_estados = df['estado'].value_counts()

# Save municipality info for map
municipio_info = df.groupby(['municipio', 'estado', 'bioma']).agg(
    latitude=('latitude', 'mean'),
    longitude=('longitude', 'mean')
).reset_index()

print("Agregando dados para o modelo...")
df_inc = df.drop(columns=['satelite', 'pais', 'data', 'latitude', 'longitude', 'area_industrial'])
df_inc['incendio'] = 1

df_inc = df_inc.groupby(['semana', 'municipio', 'estado', 'ano', 'mes', 'dia']).agg(
    incendio_total=('incendio', 'sum'),
    risco_fogo_medio=('risco_fogo', 'mean'),
    precipitacao_total=('precipitacao', 'sum'),
    dias_sem_chuva_max=('numero_dias_sem_chuva', 'max'),
    bioma=('bioma', lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
).reset_index()

# Save correlation data before adding negatives
corr_vars = ['incendio_total', 'risco_fogo_medio', 'precipitacao_total', 'dias_sem_chuva_max']
corr_matrix = df_inc[corr_vars].corr()

df_inc['teve_incendio'] = 1

print("Criando amostras negativas sintéticas...")
n = len(df_inc)
df_neg = pd.DataFrame({
    'semana': df_inc['semana'].values,
    'municipio': df_inc['municipio'].values,
    'estado': df_inc['estado'].values,
    'ano': df_inc['ano'].values,
    'mes': df_inc['mes'].values,
    'dia': df_inc['dia'].values,
    'incendio_total': np.zeros(n, dtype=int),
    'risco_fogo_medio': np.random.uniform(0, np.quantile(df_inc['risco_fogo_medio'], 0.2), size=n),
    'precipitacao_total': np.random.uniform(20, 100, size=n),
    'dias_sem_chuva_max': np.random.randint(0, 3, size=n),
    'bioma': df_inc['bioma'].values,
    'teve_incendio': np.zeros(n, dtype=int)
})
df_final = pd.concat([df_inc, df_neg], ignore_index=True)

df_final['semana_sin'] = np.sin(2 * np.pi * df_final['semana'] / 52)
df_final['semana_cos'] = np.cos(2 * np.pi * df_final['semana'] / 52)

print("Codificando variáveis categóricas...")
encoders = {}
for col in ['municipio', 'estado', 'bioma']:
    le = LabelEncoder()
    df_final[col] = le.fit_transform(df_final[col].astype(str))
    encoders[col] = le

# Save biome classes for UI dropdown
biomas_ordenados = sorted(encoders['bioma'].classes_)

print("Separando features para treino...")
feature_cols = ['mes', 'precipitacao_total', 'dias_sem_chuva_max', 'bioma', 'semana_sin', 'semana_cos']
x = df_final[feature_cols]
y = df_final['teve_incendio']

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)
x_train = x_train.astype(float)
x_test = x_test.astype(float)

print("Aplicando SMOTE...")
smote = SMOTE(random_state=42)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

print("Treinando XGBoost...")
model = xgb.XGBClassifier(
    n_estimators=500, max_depth=3, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.8, random_state=42,
    scale_pos_weight=15, reg_lambda=1, reg_alpha=0.1
)
model.fit(x_train_res, y_train_res)

y_pred = model.predict_proba(x_test)[:, 1]
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred):.4f}")
print(f"Log Loss: {log_loss(y_test, y_pred):.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("Salvando artefatos...")
joblib.dump(model, os.path.join(OUTPUT_DIR, 'xgb_model.pkl'))
joblib.dump(encoders, os.path.join(OUTPUT_DIR, 'label_encoders.pkl'))
joblib.dump(municipio_info, os.path.join(OUTPUT_DIR, 'municipio_info.pkl'))
joblib.dump(biomas_ordenados, os.path.join(OUTPUT_DIR, 'biomas_list.pkl'))
joblib.dump(feature_importance, os.path.join(OUTPUT_DIR, 'feature_importance.pkl'))
joblib.dump(corr_matrix, os.path.join(OUTPUT_DIR, 'corr_matrix.pkl'))
joblib.dump(chart_biomas, os.path.join(OUTPUT_DIR, 'chart_biomas.pkl'))
joblib.dump(chart_meses, os.path.join(OUTPUT_DIR, 'chart_meses.pkl'))
joblib.dump(chart_estados, os.path.join(OUTPUT_DIR, 'chart_estados.pkl'))
joblib.dump(feature_cols, os.path.join(OUTPUT_DIR, 'feature_cols.pkl'))

print("Gerando previsões para o mapa...")
df_map_feats = df_final.groupby('municipio').agg({
    'mes': 'last',
    'precipitacao_total': 'mean',
    'dias_sem_chuva_max': 'mean',
    'bioma': 'last',
}).reset_index()

semana_atual = pd.Timestamp.now().isocalendar().week
semana_alvo = semana_atual + 1
if semana_alvo > 52:
    semana_alvo = 1

df_map_feats['semana_sin'] = np.sin(2 * np.pi * semana_alvo / 52)
df_map_feats['semana_cos'] = np.cos(2 * np.pi * semana_alvo / 52)
df_map_feats['semana'] = semana_alvo

x_map = df_map_feats[feature_cols].astype(float)
df_map_feats['probabilidade'] = model.predict_proba(x_map)[:, 1]

# Merge with municipality info (name, state, lat, lon)
df_map_feats['municipio_nome'] = encoders['municipio'].inverse_transform(df_map_feats['municipio'])
map_final = municipio_info.merge(
    df_map_feats[['municipio', 'municipio_nome', 'probabilidade', 'semana', 'precipitacao_total', 'dias_sem_chuva_max']],
    left_on='municipio',
    right_on='municipio_nome',
    how='inner'
)

map_final = map_final.drop(columns=['municipio_nome'])
map_final = map_final.rename(columns={'municipio_x': 'municipio'})
map_final.to_csv(os.path.join(OUTPUT_DIR, 'mapa_previsoes.csv'), index=False)
print(f"Mapa salvo com {len(map_final)} municípios")

print("\nTreinamento concluído!")
print(f"Artefatos salvos em: {OUTPUT_DIR}")
