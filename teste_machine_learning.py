# Primeiro, será configurado o amboente de desenvolvimento


# Importando as bibliotecas necessárias
""" pip install goolge-generativeai - instala a API do Gemini AI"""

# importando as bibioltecas pandas, numpy, matplotlib e seaborn
import pandas as pd

# Fazendo o carregamento do dataset
df = pd.read_csv('sample_superstore.csv', encoding='latin-1')

# Fazendo a visualização das primeiras linnhas do dataset
print(df.head())

# Verificando a estrutura dos dados
print(df.info())

# Aplicando estatística descritivas iniciais
print(df.describe())
