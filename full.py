# Databricks notebook source
# MAGIC %md
# MAGIC # Pipeline de Regressão Logística com Validação Cruzada de Séries Temporais e Importância de Variáveis (PySpark)
# MAGIC
# MAGIC Este notebook demonstra um pipeline completo para uma aplicação de regressão logística usando PySpark. Ele inclui:
# MAGIC
# MAGIC 1.  **Geração de Dados de Exemplo:** Para tornar o notebook executável imediatamente.
# MAGIC 2.  **Divisão de Dados por Período Anual:** Implementação de validação cruzada de séries temporais.
# MAGIC 3.  **Pré-processamento de Dados:** Indexação de strings, montagem de vetores e escalonamento.
# MAGIC 4.  **Treinamento do Modelo:** Regressão Logística.
# MAGIC 5.  **Avaliação do Modelo:** Usando AUC, AUPRC, Acurácia, Precisão, Recall e F1-Score.
# MAGIC 6.  **Teste de Importância de Variáveis:** Coeficientes do modelo, importância por permutação, correlação e teste Qui-Quadrado.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Inicializar a SparkSession

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator # Importar MulticlassClassificationEvaluator
from pyspark.sql.functions import year, month, col, to_date, lit, rand, when, ceil
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.preprocessing import StandardScaler as SKStandardScaler
from sklearn.preprocessing import OneHotEncoder as SKOneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SKPipeline
import numpy as np

# Inicializar a SparkSession
spark = SparkSession.builder.appName("LogisticRegressionTimeSeriesCV").getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Geração de Dados de Exemplo (Substitua por seus dados reais)
# MAGIC
# MAGIC Esta seção gera um DataFrame de exemplo. **Para usar seus próprios dados, substitua esta seção pelo carregamento do seu arquivo.**
# MAGIC
# MAGIC O DataFrame de exemplo terá as seguintes colunas:
# MAGIC - `id`: Identificador único.
# MAGIC - `periodo`: Mês e ano no formato `YYYYMM` (ex: 202301, 202302).
# MAGIC - `feature_num_1`: Feature numérica.
# MAGIC - `feature_num_2`: Feature numérica.
# MAGIC - `feature_cat_1`: Feature categórica (A, B, C).
# MAGIC - `feature_cat_2`: Feature categórica (X, Y).
# MAGIC - `label`: Variável alvo binária (0 ou 1).

# COMMAND ----------

# Gerar dados de exemplo
num_records_per_month = 100
data_rows = []

for year_val in range(2023, 2024): # Apenas o ano de 2023
    for month_val in range(1, 13): # Meses de 1 a 12
        periodo_str = f"{year_val}{month_val:02d}"
        for i in range(num_records_per_month):
            feature_num_1 = rand() * 100
            feature_num_2 = rand() * 50 + (month_val * 2) # Adiciona uma tendência temporal
            feature_cat_1 = np.random.choice(['A', 'B', 'C'], p=[0.5, 0.3, 0.2])
            feature_cat_2 = np.random.choice(['X', 'Y'], p=[0.7, 0.3])
            # Criar um rótulo com alguma dependência das features e do tempo
            label = 0
            if feature_num_1 > 70 and feature_num_2 > 40 and feature_cat_1 == 'A':
                label = 1
            if month_val >= 10 and rand() > 0.6: # Aumenta a chance de label=1 nos últimos meses
                label = 1
            elif rand() > 0.9: # Pequena chance aleatória de label=1
                label = 1

            data_rows.append((i + (month_val - 1) * num_records_per_month, int(periodo_str), feature_num_1, feature_num_2, feature_cat_1, feature_cat_2, label))

schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("periodo", IntegerType(), True),
    StructField("feature_num_1", DoubleType(), True),
    StructField("feature_num_2", DoubleType(), True),
    StructField("feature_cat_1", StringType(), True),
    StructField("feature_cat_2", StringType(), True),
    StructField("label", IntegerType(), True)
])

data = spark.createDataFrame(data_rows, schema=schema)

# 2.1. Preparar a coluna de período
# Supondo que sua coluna de período esteja no formato `YYYYMM`, vamos criar uma coluna de data para facilitar a ordenação
data = data.withColumn("ano", year(to_date(col("periodo").cast("string"), "yyyyMM")))
data = data.withColumn("mes", month(to_date(col("periodo").cast("string"), "yyyyMM")))
data = data.orderBy(col("ano"), col("mes")) # Ordenar por período

print("Dados de exemplo gerados:")
data.printSchema()
data.show(5)
print(f"Total de registros: {data.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Definir Parâmetros e Folds para Validação Cruzada de Séries Temporais

# COMMAND ----------

period_column = "periodo"  # Coluna que contém o período (ex: 202301)
label_col = "label"      # Coluna da variável alvo

# Obter a lista de períodos únicos e ordenados
periods = [row[0] for row in data.select(period_column).distinct().orderBy(period_column).collect()]
print(f"Períodos disponíveis: {periods}")

# Parâmetros para a validação cruzada de séries temporais
# Ex: 3 folds, cada fold de teste com 3 meses.
# O primeiro fold de teste começará no 4º período (índice 3, se os períodos são 0-indexed)
num_folds = 3
fold_length = 3 # Número de períodos (meses) em cada fold de teste
start_fold_index = 3 # Índice do primeiro período a ser usado como teste no primeiro fold (ex: 202304)

folds = []
for i in range(num_folds):
    test_start_index = start_fold_index + i * fold_length
    test_end_index = test_start_index + fold_length

    if test_end_index > len(periods):
        break # Parar se não houver períodos suficientes para o fold completo

    test_periods = periods[test_start_index:test_end_index]
    train_periods = periods[:test_start_index]

    if train_periods and test_periods:
        folds.append({'train': train_periods, 'test': test_periods})

print("\nFolds para Validação Cruzada:")
if not folds:
    print("Não foi possível criar folds de validação cruzada com os parâmetros definidos. Ajuste 'start_fold_index', 'num_folds' ou 'fold_length'.")
    spark.stop()
    dbutils.notebook.exit()
else:
    for i, fold in enumerate(folds):
        print(f"Fold {i+1}: Treino ({len(fold['train'])} períodos) - {fold['train']}, Teste ({len(fold['test'])} períodos) - {fold['test']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Pré-processamento dos Dados

# COMMAND ----------

# 4.1. Identificar colunas categóricas e numéricas (excluindo colunas de período e rótulo)
feature_cols = [col_name for col_name in data.columns if col_name not in [period_column, 'ano', 'mes', label_col, 'id']] # 'id' também é excluído
categorical_cols = [col_name for col_name, dtype in data.dtypes if dtype == "string" and col_name in feature_cols]
numerical_cols = [col_name for col_name, dtype in data.dtypes if dtype != "string" and col_name in feature_cols]

print(f"Colunas de features: {feature_cols}")
print(f"Colunas categóricas: {categorical_cols}")
print(f"Colunas numéricas: {numerical_cols}")
print(f"Coluna de rótulo: {label_col}")

# 4.2. Indexação de colunas categóricas
indexers = [StringIndexer(inputCol=col_name, outputCol=col_name + "_index", handleInvalid="keep") for col_name in categorical_cols]

# 4.3. Montagem das features numéricas e indexadas em um vetor
assembler_inputs = [col_name + "_index" for col_name in categorical_cols] + numerical_cols
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="raw_features", handleInvalid="skip")

# 4.4. Normalização/Padronização das features
scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Treinamento do Modelo de Regressão Logística e Validação Cruzada

# COMMAND ----------

# Criar o objeto do modelo de Regressão Logística
lr = LogisticRegression(labelCol=label_col, featuresCol="features")

# Configurar os avaliadores
binary_evaluator = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction")
multiclass_evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction")

results = []
trained_models = [] # Para armazenar os modelos treinados de cada fold (útil para inspeção)

for i, fold_info in enumerate(folds):
    train_periods = fold_info['train']
    test_periods = fold_info['test']

    print(f"\n===== Rodando Fold {i+1} =====")
    print(f"  Treinando nos períodos: {train_periods}")
    print(f"  Testando nos períodos: {test_periods}")

    # Filtrar os dados para o fold atual
    train_fold_data = data.filter(col(period_column).isin(train_periods))
    test_fold_data = data.filter(col(period_column).isin(test_periods))

    if train_fold_data.isEmpty() or test_fold_data.isEmpty():
        print(f"  Aviso: Conjunto de treino ou teste vazio para o Fold {i+1}. Pulando este fold.")
        continue

    # Criar o pipeline para este fold
    pipeline = Pipeline(stages=indexers + [assembler, scaler, lr])

    # Treinar o modelo
    model = pipeline.fit(train_fold_data)
    trained_models.append(model) # Armazena o modelo treinado

    # Fazer previsões no conjunto de teste
    predictions = model.transform(test_fold_data)

    # Avaliar o modelo com diversas métricas
    auc = binary_evaluator.evaluate(predictions, {binary_evaluator.metricName: "areaUnderROC"})
    auprc = binary_evaluator.evaluate(predictions, {binary_evaluator.metricName: "areaUnderPR"})
    accuracy = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "accuracy"})
    f1_score = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "f1"})
    precision = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "weightedPrecision"})
    recall = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "weightedRecall"})

    print(f"  AUC no Fold {i+1}: {auc:.4f}")
    print(f"  AUPRC no Fold {i+1}: {auprc:.4f}")
    print(f"  Acurácia no Fold {i+1}: {accuracy:.4f}")
    print(f"  Precisão (Ponderada) no Fold {i+1}: {precision:.4f}")
    print(f"  Recall (Ponderado) no Fold {i+1}: {recall:.4f}")
    print(f"  F1-Score (Ponderado) no Fold {i+1}: {f1_score:.4f}")

    results.append({
        'fold': i+1,
        'auc': auc,
        'auprc': auprc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    })

# Calcular a média e o desvio padrão do desempenho para cada métrica
if results:
    print(f"\n===== Resultados Agregados da Validação Cruzada =====")
    
    # Coletar todas as métricas para cálculo de média e desvio padrão
    all_aucs = [r['auc'] for r in results]
    all_auprcs = [r['auprc'] for r in results]
    all_accuracies = [r['accuracy'] for r in results]
    all_precisions = [r['precision'] for r in results]
    all_recalls = [r['recall'] for r in results]
    all_f1_scores = [r['f1_score'] for r in results]

    print(f"Média da AUC: {np.mean(all_aucs):.4f} (Std: {np.std(all_aucs):.4f})")
    print(f"Média da AUPRC: {np.mean(all_auprcs):.4f} (Std: {np.std(all_auprcs):.4f})")
    print(f"Média da Acurácia: {np.mean(all_accuracies):.4f} (Std: {np.std(all_accuracies):.4f})")
    print(f"Média da Precisão (Ponderada): {np.mean(all_precisions):.4f} (Std: {np.std(all_precisions):.4f})")
    print(f"Média do Recall (Ponderado): {np.mean(all_recalls):.4f} (Std: {np.std(all_recalls):.4f})")
    print(f"Média do F1-Score (Ponderado): {np.mean(all_f1_scores):.4f} (Std: {np.std(all_f1_scores):.4f})")

else:
    print("\nNenhum resultado de validação cruzada foi gerado.")

# Para a análise de importância de variáveis, usaremos o modelo treinado no ÚLTIMO fold para demonstração.
# Em um cenário real, você treinaria um modelo final em todos os dados de treinamento disponíveis após a validação.
if trained_models:
    final_model_for_importance = trained_models[-1]
    print("\nUtilizando o modelo do ÚLTIMO fold para análise de importância de variáveis.")
else:
    print("\nNenhum modelo foi treinado para análise de importância de variáveis.")
    spark.stop()
    dbutils.notebook.exit()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Testando a Importância de Variáveis
# MAGIC
# MAGIC As análises de importância de variáveis abaixo são baseadas no modelo treinado no **último fold** da validação cruzada.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1. Coeficientes da Regressão Logística (Intrínseco ao Modelo)

# COMMAND ----------

# Obter o modelo de regressão logística treinado do pipeline
trained_lr_model_stage = final_model_for_importance.stages[-1]

# Obter os coeficientes
coefficients = trained_lr_model_stage.coefficients

# Obter os nomes das features (após o VectorAssembler)
# O VectorAssembler está na penúltima posição do pipeline, antes do scaler e do LR
vector_assembler_stage = final_model_for_importance.stages[len(indexers)]
feature_names_assembled = vector_assembler_stage.getInputCols()

# Criar uma lista de tuplas com nome da feature e coeficiente
feature_importance_lr = zip(feature_names_assembled, coefficients.toArray().tolist())

# Ordenar por magnitude do coeficiente (valor absoluto)
sorted_importance_lr = sorted(feature_importance_lr, key=lambda x: abs(x[1]), reverse=True)

print("Importância das Variáveis (Regressão Logística - Coeficientes):")
for feature, importance in sorted_importance_lr:
    print(f"- {feature}: {importance:.4f}")

# **Observações:**
# - A magnitude do coeficiente indica a força da relação.
# - O sinal indica a direção da relação (positiva ou negativa com a probabilidade da classe positiva).
# - Variáveis categóricas são representadas por seus índices (ex: `nome_da_coluna_index`).

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2. Importância por Permutação (Feature Permutation Importance) - Abordagem com Pandas e Scikit-learn
# MAGIC
# MAGIC **Observação:** Esta abordagem converte uma amostra dos dados de teste para Pandas e usa `scikit-learn`. Para conjuntos de dados muito grandes, isso pode ser intensivo em memória.

# COMMAND ----------

# Usar os dados de teste do último fold para a permutação
last_test_periods = folds[-1]['test']
pandas_test_df = data.filter(col(period_column).isin(last_test_periods)).toPandas()

# Separar features e rótulo
X_test_perm = pandas_test_df.drop(columns=[label_col, period_column, 'ano', 'mes', 'id'])
y_test_perm = pandas_test_df[label_col]

# Definir pré-processadores para colunas categóricas e numéricas (semelhante ao pipeline PySpark)
# Replicando as transformações do PySpark para o scikit-learn
numerical_transformer = SKStandardScaler()
categorical_transformer = SKPipeline(steps=[
    ('onehot', SKOneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='passthrough' # Manter outras colunas se houver
)

# Criar e treinar um pipeline scikit-learn
# O modelo scikit-learn será treinado APENAS para calcular a importância por permutação
# Não estamos usando este modelo para previsões gerais, apenas para a análise de importância
sk_pipeline = SKPipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', SKLogisticRegression(random_state=42, solver='liblinear'))]) # solver='liblinear' para pequenas amostras

# Treinar o pipeline scikit-learn nos dados de TREINO do último fold para consistência
last_train_periods = folds[-1]['train']
pandas_train_df_for_sk = data.filter(col(period_column).isin(last_train_periods)).toPandas()
X_train_sk = pandas_train_df_for_sk.drop(columns=[label_col, period_column, 'ano', 'mes', 'id'])
y_train_sk = pandas_train_df_for_sk[label_col]

sk_pipeline.fit(X_train_sk, y_train_sk)

# Obter os nomes das features após o pré-processamento do scikit-learn
# Isso é importante para mapear a importância de volta aos nomes originais
feature_names_out = sk_pipeline.named_steps['preprocessor'].get_feature_names_out()

# Realizar a importância por permutação
perm_importance = permutation_importance(sk_pipeline, X_test_perm, y_test_perm, n_repeats=10, random_state=42)

# Obter as importâncias
importances = perm_importance.importances_mean

# Criar uma lista de tuplas com nome da feature e importância
feature_importance_perm = zip(feature_names_out, importances.tolist())

# Ordenar por magnitude da importância
sorted_importance_perm = sorted(feature_importance_perm, key=lambda x: x[1], reverse=True)

print("\nImportância das Variáveis (Permutação - Amostra do Teste do Último Fold):")
for feature, importance in sorted_importance_perm:
    print(f"- {feature}: {importance:.4f}")

# **Interpretação:**
# - Valores positivos de importância indicam que a feature contribuiu para o desempenho do modelo.
# - Valores negativos ou próximos de zero sugerem que a feature pode não ser informativa ou pode até adicionar ruído.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.3. Correlação (para variáveis numéricas)

# COMMAND ----------

# Usar os dados de treinamento do último fold para a correlação
train_data_for_corr = data.filter(col(period_column).isin(folds[-1]['train']))

print(f"\nCorrelação das Variáveis Numéricas com a Variável de Rótulo (Conjunto de Treinamento do Último Fold):")
for num_col in numerical_cols:
    try:
        correlation = train_data_for_corr.stat.corr(num_col, label_col)
        print(f"- Correlação entre {num_col} e {label_col}: {correlation:.4f}")
    except Exception as e:
        print(f"- Não foi possível calcular a correlação entre {num_col} e {label_col}: {e}")

# **Observações:**
# - A correlação mede a relação linear entre duas variáveis numéricas.
# - Valores próximos de +1 indicam uma correlação positiva forte, -1 uma correlação negativa forte e 0 nenhuma correlação linear.
# - Este método considera apenas a relação univariada.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.4. Teste Qui-Quadrado (para variáveis categóricas e alvo categórico)

# COMMAND ----------

from pyspark.ml.stat import ChiSquareTest
from pyspark.ml.feature import OneHotEncoder

# Usar os dados de treinamento do último fold para o teste Qui-Quadrado
train_data_for_chi2 = data.filter(col(period_column).isin(folds[-1]['train']))

if categorical_cols:
    print(f"\nTeste Qui-Quadrado para Variáveis Categóricas (Conjunto de Treinamento do Último Fold):")
    for cat_col in categorical_cols:
        try:
            # Indexar a coluna categórica
            indexer = StringIndexer(inputCol=cat_col, outputCol=cat_col + "_index_chi", handleInvalid="keep").fit(train_data_for_chi2)
            indexed_data = indexer.transform(train_data_for_chi2)

            # Executar o teste qui-quadrado
            # Nota: ChiSquareTest espera um vetor de features. Para uma única coluna categórica,
            # podemos usar o ChiSquareTest.test diretamente com a coluna indexada.
            # Se você tivesse múltiplas features categóricas para testar de uma vez,
            # precisaria de um VectorAssembler para elas antes do ChiSquareTest.
            chi_squared_test = ChiSquareTest.test(indexed_data, cat_col + "_index_chi", label_col)
            result = chi_squared_test.head()
            print(f"- Teste Qui-Quadrado para {cat_col}: Chi-squared = {result.statistic:.4f}, p-value = {result.pValue:.4f}")
        except Exception as e:
            print(f"- Não foi possível realizar o teste qui-quadrado para {cat_col}: {e}")
else:
    print("\nNão há colunas categóricas para realizar o teste qui-quadrado.")

# **Interpretação:**
# - Um p-value baixo (geralmente abaixo de 0.05) sugere que há uma dependência estatisticamente significativa entre a variável categórica e a variável de rótulo.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Parar a SparkSession

# COMMAND ----------

spark.stop()
