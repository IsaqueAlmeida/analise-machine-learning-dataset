""" Limpeza e Tratamento dos Dados """
import teste_machine_learning as tml
import pandas as pd
# import numpy as np

df = tml.df

# Conversão de coluna de datas

df['Order  Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

# Nesse momento, checando os valores nulos
print(df.isnull().sum())

# Verificando duplicidades
print(df.duplicated().sum())
