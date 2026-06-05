


def tratar_dados(df):
    import pandas as pd
    import numpy as np

    #Conversão dos tipos de dados para os corretos
    df['data'] = pd.to_datetime(df['data_pas'], errors='coerce')
    df['mes'] = df['data'].dt.month
    df['ano'] = df['data'].dt.year
    df['semana'] = df['data'].dt.isocalendar().week
    df['dia'] = df['data'].dt.day

    #Tratamento dos dados, convertendo os valores negativos para NaN e preenchendo com a mediana
    df['id_area_industrial'] =df['id_area_industrial'].apply(lambda x: 1 if x > 0 else 0)
    df['numero_dias_sem_chuva'] = df['numero_dias_sem_chuva'].apply(lambda x: np.nan if x < 0 else x)
    df['risco_fogo'] = df['risco_fogo'].apply(lambda x: np.nan if x < 0 else x)
    df['numero_dias_sem_chuva'] = df['numero_dias_sem_chuva'].fillna(df['numero_dias_sem_chuva'].median())
    df['risco_fogo'] = df['risco_fogo'].fillna(df['risco_fogo'].median())

    #Remoção de colunas desnecessárias
    df = df.rename(columns={'id_area_industrial': 'area_industrial'})

    #Removendo colunas
    df = df.drop(columns=['data_pas'])

    df_tratado = df

    return df_tratado