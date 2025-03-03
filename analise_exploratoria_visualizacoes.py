""" Análise Exploratória dos Dados (EDA) """
import teste_machine_learning as tml
import seaborn as sns
import matplotlib.pyplot as plt

df = tml.df

# Distribuição das vendas
plt.figure(figsize=(8, 5))
sns.histplot(df['Sales'], kde=True)
plt.title('Distribuição de Vendas')
plt.xlabel('Sales')
plt.ylabel('Frequência')
plt.show()

# Relação entre Sales e Profit
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Sales', y='Profit', data=df)
plt.title('Relação entre Vendas e Lucros')
plt.xlabel('Sales')
plt.ylabel('Profit')
plt.show()
