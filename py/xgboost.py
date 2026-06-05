


def xgboost(df_tratado):

    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE
    import xgboost as xgb

    df_incendios = df_tratado.drop(columns= ['satelite','pais','data','latitude','longitude','area_industrial','risco_fogo'])
    df_incendios['incendio'] = 1

    df_incendios['precipitacao_total'] = df_incendios['precipitacao'].fillna(0)

    df_incendios = df_incendios.groupby(['semana','municipio','estado','ano','mes','dia']).agg(
        incendio_total=('incendio','sum'),
        precipitacao_total=('precipitacao','sum'),
        dias_sem_chuva_max=('numero_dias_sem_chuva','max'),
        bioma=('bioma',  pd.Series.mode) 
        ).reset_index()

    df_incendios['teve_incendio'] = 1
    negativos = []


    n = len(df_incendios)

    df_negativos = pd.DataFrame({
        'semana': df_incendios['semana'].values,
        'municipio': df_incendios['municipio'].values,
        'estado': df_incendios['estado'].values,
        'ano': df_incendios['ano'].values,
        'mes': df_incendios['mes'].values,
        'dia': df_incendios['dia'].values,
        'incendio_total': np.zeros(n, dtype=int),
        'precipitacao_total': np.random.uniform(20, 100, size=n),
        'dias_sem_chuva_max': np.random.randint(0, 3, size=n),
        'bioma': df_incendios['bioma'].values,
        'teve_incendio': np.zeros(n, dtype=int)
        })

    # Concatenar positivos e negativos
    df_xgb = pd.concat([df_incendios, df_negativos], ignore_index=True)
    df_xgb['semana_sin'] = np.sin(2 * np.pi * df_xgb['semana'] / 52)
    df_xgb['semana_cos'] = np.cos(2 * np.pi * df_xgb['semana'] / 52)

    for col in ['municipio', 'estado', 'bioma']:
        le = LabelEncoder()
        df_xgb[col] = le.fit_transform(df_xgb[col].astype(str))
    
    x = df_xgb.drop(columns=['teve_incendio','dia','estado','semana','incendio_total','municipio','ano','risco_fogo_medio'])

    y = df_xgb['teve_incendio']

    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y,test_size=0.2, random_state=42)
    x_train = x_train.astype(float)

    smote = SMOTE(random_state=42)
    x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

    xgb_clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.8,
        random_state=42,
        scale_pos_weight=15,
        reg_lambda=1,  # L2 regularization
        reg_alpha=0.1   # L1 regularization  
        )

    xgb_clf.fit(x_train_res, y_train_res)

    y_pred_proba = xgb_clf.predict_proba(x_test)[:, 1]

    
    