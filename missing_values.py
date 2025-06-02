import pandas as pd
import numpy as np

def avaliar_e_tratar_missing(df):
    """
    Avalia e trata missing values em um DataFrame.

    Args:
        df (pd.DataFrame): O DataFrame de entrada.

    Returns:
        pd.DataFrame: O DataFrame após o tratamento de missing values.
    """

    # 1. Identificar a porcentagem de missing values por coluna
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    print("Porcentagem de Missing Values por Coluna:")
    print(missing_percentage[missing_percentage > 0])  # Exibe apenas colunas com missing values

    # 2. Decidir como tratar (exemplo: remover colunas com muitos missing, imputar outros)
    # Exemplo: Remover colunas com mais de 50% de missing
    threshold_remover_colunas = 50
    colunas_para_remover = missing_percentage[missing_percentage > threshold_remover_colunas].index
    df = df.drop(columns=colunas_para_remover)
    print(f"\nColunas removidas ({len(colunas_para_remover)}): {colunas_para_remover}")

    # Exemplo: Imputar missing values restantes com a mediana (robusto a outliers)
    colunas_para_imputar = df.columns[df.isnull().any()].tolist()
    for coluna in colunas_para_imputar:
        df[coluna] = df[coluna].fillna(df[coluna].median())
    print(f"\nMissing values imputados com mediana nas colunas: {colunas_para_imputar}")

    # 3. Verificar se ainda há missing values
    print("\nMissing values restantes:")
    print(df.isnull().sum().sum())

    return df

# Exemplo de uso:
# Suponha que seus dados estão em um DataFrame chamado 'df'
# df = pd.read_csv('seu_arquivo.csv')
# df_tratado = avaliar_e_tratar_missing(df)
