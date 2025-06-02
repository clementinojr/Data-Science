def avaliar_correlacao(df, threshold_correlacao=0.8):
    """
    Avalia a correlação entre as features em um DataFrame.

    Args:
        df (pd.DataFrame): O DataFrame de entrada.
        threshold_correlacao (float): O threshold para a correlação.

    Returns:
        list: Uma lista de pares de features altamente correlacionadas.
    """

    correlacao_matrix = df.corr().abs()
    upper_tri = correlacao_matrix.where(np.triu(np.ones(correlacao_matrix.shape),k=1).astype(bool))
    colunas_para_remover = [coluna for coluna in upper_tri.columns if any(upper_tri[coluna] > threshold_correlacao)]
    pares_correlacionados = []
    for coluna in colunas_para_remover:
        colunas_correlacionadas = upper_tri.index[upper_tri[coluna] > threshold_correlacao].tolist()
        for outra_coluna in colunas_correlacionadas:
            pares_correlacionados.append((coluna, outra_coluna, correlacao_matrix[coluna][outra_coluna]))
    
    return pares_correlacionados

# Exemplo de uso:
# pares_correlacionados = avaliar_correlacao(df_tratado.drop(columns=['target']))
# print("Pares de features altamente correlacionadas:")
# for par in pares_correlacionados:
#     print(f"Coluna 1: {par[0]}, Coluna 2: {par[1]}, Correlação: {par[2]:.2f}")

def remover_correlacao(df, threshold_correlacao=0.8):
    """
    Remove features altamente correlacionadas.

    Args:
        df (pd.DataFrame): O DataFrame de entrada.
        threshold_correlacao (float): O threshold para a correlação.

    Returns:
        pd.DataFrame: O DataFrame após a remoção de features.
    """
    df_sem_correlacao = df.copy()
    correlacao_matrix = df_sem_correlacao.corr().abs()
    upper_tri = correlacao_matrix.where(np.triu(np.ones(correlacao_matrix.shape),k=1).astype(bool))
    colunas_para_remover = [coluna for coluna in upper_tri.columns if any(upper_tri[coluna] > threshold_correlacao)]
    df_sem_correlacao = df_sem_correlacao.drop(columns=colunas_para_remover)
    print(f"Colunas removidas por alta correlação: {colunas_para_remover}")
    return df_sem_correlacao

# Exemplo de uso:
# df_sem_correlacao = remover_correlacao(df_tratado.drop(columns=['target']), threshold_correlacao=0.8)
