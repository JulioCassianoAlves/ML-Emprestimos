# Trabalho de Ciência de Dados
# Análise de Dados e Modelagem de Machine Learning
# Autor: Júlio Cassiano Alves
# Data: 25/11/2025

import os
# Configuração para evitar erros de multiprocessamento e warnings do OpenBLAS
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Define limite de cores para joblib evitar erro do wmic
os.environ['OPENBLAS_NUM_THREADS'] = '1' # Evita warning do OpenBLAS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Configuração de pastas
PASTA_ANALISE = 'analise'
PASTA_RESULTADOS = 'resultados'

os.makedirs(PASTA_ANALISE, exist_ok=True)
os.makedirs(PASTA_RESULTADOS, exist_ok=True)

def carregar_dados(caminho):
  print(f"Carregando dados de {caminho}...")
  
  df = pd.read_csv(caminho)
  
  # Retorna dados básicos sobre o arquivo/dataset carregado
  # qtd colunas, qtd registros não nulos, tipo das colunas
  print(df.info())
  print(df.head())
  
  # Retorna estatistica básicas dos campos do dadaset
  print(df.describe())
  
  return df

def tratamento_dados(df):
  print("Iniciando tratamento de dados...")

  # remover os dados duplicados
  df.drop_duplicates(inplace=True)
  
  # Verificar nulos
  if df.isnull().sum().sum() > 0:
    print("Valores nulos encontrados. Tratando...")
    df.dropna(inplace=True) # Simples drop para este caso, dado o overview dizer que é completo
  
  # Remover customer_id
  df.drop('customer_id', axis=1, inplace=True)

  return df

def analise_grafica(df):
  print("Gerando gráficos de análise...")
  
  # Distribuição da variável alvo
  plt.figure(figsize=(6, 4))
  sns.countplot(x='loan_status', data=df)
  plt.title('Distribuição de Aprovação de Empréstimos')
  plt.savefig(os.path.join(PASTA_ANALISE, 'distribuicao_target.png'))
  plt.close()
  
  # Matriz de Correlação
  plt.figure(figsize=(12, 10))
  numeric_df = df.select_dtypes(include=[np.number])
  sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
  plt.title('Matriz de Correlação')
  plt.savefig(os.path.join(PASTA_ANALISE, 'correlacao.png'))
  plt.close()

  # Histogramas para variáveis numéricas
  n = len(numeric_df.columns)
  linhas = (n + 1) // 2
  plt.figure(figsize=(12, linhas * 4))
  for i, col in enumerate(numeric_df.columns, 1):
    plt.subplot(linhas, 2, i)
    sns.histplot(numeric_df[col])
    plt.title(f"Histograma de {col}")
  plt.tight_layout()
  plt.savefig(os.path.join(PASTA_ANALISE, 'histogramas_numericos.png'))
  plt.close()
  
  # Boxplots para algumas variáveis importantes vs Target
  cols_to_plot = ['annual_income', 'credit_score', 'debt_to_income_ratio', 'loan_amount']
  for col in cols_to_plot:
    if col in df.columns:
      plt.figure(figsize=(8, 6))
      sns.boxplot(x='loan_status', y=col, data=df)
      plt.title(f'{col} vs Loan Status')
      plt.savefig(os.path.join(PASTA_ANALISE, f'boxplot_{col}.png'))
      plt.close()

def separar_dados(df):
  print("Separando dados em Treino, Teste e Validação...")
  X = df.drop('loan_status', axis=1)
  y = df['loan_status']
  
  # 70% Treino, 30% Temp
  X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
  
  # 50% do Temp para Teste (15% total), 50% do Temp para Validação (15% total)
  X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
  
  print(f"Treino: {X_train.shape}, Teste: {X_test.shape}, Validação: {X_val.shape}")
  return X_train, X_test, X_val, y_train, y_test, y_val

def treinar_avaliar(X_train, y_train, X_test, y_test):
  print("Iniciando treinamento e avaliação...")
  
  # Pré-processamento
  categorical_cols = X_train.select_dtypes(include=['object']).columns
  numerical_cols = X_train.select_dtypes(include=['number']).columns
  
  preprocessor = ColumnTransformer(
    transformers=[
      ('num', StandardScaler(), numerical_cols),
      ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])
  
  modelos = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Gaussian Naive Bayes': GaussianNB(),
    'Multi-layer Perceptron': MLPClassifier(max_iter=1000, random_state=42),
    'KNeighbors': KNeighborsClassifier(),
    'SVC': SVC(probability=True, random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
  }
  
  resultados = []
  melhor_modelo = None
  melhor_f1 = 0
  nome_melhor_modelo = ""
  
  for nome, modelo in modelos.items():
    print(f"Treinando {nome}...")
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', modelo)])
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    
    print(f"  -> Acc: {acc:.4f}, ROC: {roc:.4f}, F1: {f1:.4f}")
    
    resultados.append({
      'Modelo': nome,
      'Accuracy': acc,
      'ROC-AUC': roc,
      'F1-Score': f1
    })
    
    if f1 > melhor_f1:
      melhor_f1 = f1
      melhor_modelo = clf
      nome_melhor_modelo = nome
          
  # Salvar resultados
  df_resultados = pd.DataFrame(resultados)
  df_resultados.to_csv(os.path.join(PASTA_RESULTADOS, 'metricas_modelos.csv'), index=False)
  print("\nResumo dos Resultados:")
  print(df_resultados)
  
  # Salvar melhor modelo
  print(f"\nMelhor modelo: {nome_melhor_modelo} com F1: {melhor_f1:.4f}")
  with open(os.path.join(PASTA_RESULTADOS, 'modelo_final.pkl'), 'wb') as f:
    pickle.dump(melhor_modelo, f)
      
  return df_resultados

def main():
  arquivo_dados = 'Emprestimos_2025.csv'
  if not os.path.exists(arquivo_dados):
    print(f"Arquivo {arquivo_dados} não encontrado!")
    return

  df = carregar_dados(arquivo_dados)
  df = tratamento_dados(df)
  analise_grafica(df)
  X_train, X_test, X_val, y_train, y_test, y_val = separar_dados(df)
  treinar_avaliar(X_train, y_train, X_test, y_test)
  print("\nProcesso concluído com sucesso!")

if __name__ == "__main__":
  main()
