import pandas as pd
import os

def ler_csvs_inbe():
    pasta = r'C:\Users\vitor\OneDrive\Documentos\Github\previsao_incendio\dados'
    arquivos = os.listdir(pasta)
    dfs = []

    for arquivo in arquivos:
        if arquivo.startswith('focos_br') and arquivo.endswith('.csv'):
            caminho = os.path.join(pasta, arquivo)
            df = pd.read_csv(caminho, sep=",",  encoding='latin-1')
            dfs.append(df)

    df_final = pd.concat(dfs, ignore_index=True)
    print('Importação feita com sucesso!')
    return df_final



ler_csvs_inbe()

