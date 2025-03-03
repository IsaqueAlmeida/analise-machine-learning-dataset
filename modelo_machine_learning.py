""" Desenvolvimento do Modelo Machine Learning"""

import teste_machine_learning as tml
# divide os dados em conjuntos de treinamento e teste
from sklearn.model_selection import train_test_split
# Implementação do modelo de Regressão Linear
from sklearn.linear_model import LinearRegression
# Calcula métrica para avaliar o modelo
from sklearn.metrics import mean_absolute_error

df = tml.df

# Selecionando as features e target: ajustando conforme necessário
feature = df[['Quantity', 'Discount']]  # quantidade e descontos
target = df[['Sales', 'Profit']]  # vendas e lucros

# Dividindo os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(feature, target,
                                                    test_size=0.2,
                                                    random_state=42
                                                    )

# Treinando o modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Avaliando o modelo
predictions = model.predict(X_test)
print(f'MAE: {mean_absolute_error(y_test, predictions):.2f}')
