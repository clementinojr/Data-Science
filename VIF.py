from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def calcular_vif(df):
    """
    Calcula o Variance Inflation Factor (VIF) para cada feature em um DataFrame.

    Args:
        df (pd.DataFrame): O DataFrame de entrada.

    Returns:
        pd.DataFrame: Um DataFrame com os VIFs.
    """

    # Adicionar uma constante para o cálculo do VIF
    df_vif = add_constant(df)
    vif_data = pd.DataFrame()
    vif_data["feature"] = df_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(df_vif.values, i) for i in range(df_vif.shape[1])]

    return vif_data.sort_values(by="VIF", ascending=False)

# Exemplo de uso:
# Suponha que seu DataFrame tratado é 'df_tratado'
# vif_resultado = calcular_vif(df_tratado.drop(columns=['target']))  # Remova a coluna target
# print(vif_resultado)

def remover_alta_multicolinearidade(df, threshold_vif=10):
    """
    Remove features com alta multicolinearidade (VIF acima de um threshold).

    Args:
        df (pd.DataFrame): O DataFrame de entrada.
        threshold_vif (float): O threshold para o VIF.

    Returns:
        pd.DataFrame: O DataFrame após a remoção de features.
    """

    df_sem_multicolinearidade = df.copy()
    while True:
        vif_resultado = calcular_vif(df_sem_multicolinearidade.drop(columns=['target'], errors='ignore')) # 'target' caso exista
        maiores_vif = vif_resultado[vif_resultado['VIF'] > threshold_vif]
        if maiores_vif.empty:
            break
        feature_para_remover = maiores_vif.iloc[0, 0]
        if feature_para_remover == 'const':
            feature_para_remover = maiores_vif.iloc[1, 0] # Remover a segunda se a constante for a primeira
        df_sem_multicolinearidade = df_sem_multicolinearidade.drop(columns=[feature_para_remover])
        print(f"Removendo feature: {feature_para_remover}, VIF: {maiores_vif[maiores_vif['feature'] == feature_para_remover]['VIF'].values[0]}")
    return df_sem_multicolinearidade

# Exemplo de uso
# df_sem_multicolinearidade = remover_alta_multicolinearidade(df_tratado, threshold_vif=10)
