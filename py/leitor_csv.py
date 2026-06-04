import pandas as pd
import os

def ler_csvs_inbe():
    pasta = r'C:\Users\vitor\OneDrive\Documentos\Github\previsao_incendio\dados'
    arquivos = os.listdir(pasta)
    dfs = []

    for arquivo in arquivos:
        if arquivo.startswith('focos_br') and arquivo.endswith('.csv'):
            caminho = os.path.join(pasta, arquivo)
            
            # Lê em pedaços de 100 mil linhas
            for chunk in pd.read_csv(caminho, sep=',', encoding='utf-8', chunksize=100000):
                dfs.append(chunk)

    df = pd.concat(dfs, ignore_index=True)
    print("Arquivos carregados com sucesso!")
    return df
