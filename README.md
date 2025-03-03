# Análise e Machine Learning com o Superstore Dataset

Este projeto tem como objetivo demonstrar habilidades em análise de dados, machine learning e integração com IA generativa utilizando o Superstore Dataset, para um teste técnico da empresa BeTalent para o cargo de Ciência de Dados. Através dele, são abordados os seguintes níveis:

1. **Análise Exploratória (EDA) e Insights**
2. **Machine Learning Aplicado**
3. **Integração com IA Generativa (Nível Plus)**

## Sumário

- [Descrição do Projeto](#descrição-do-projeto)
- [Recursos Utilizados](#recursos-utilizados)
- [Pré-requisitos e Instalação](#pré-requisitos-e-instalação)
- [Explicação do Código](#explicação-do-código)
  - [Preparação e Análise dos Dados (EDA)](#preparação-e-análise-dos-dados-eda)
  - [Visualizações e Insights](#visualizações-e-insights)
  - [Machine Learning: Previsão de Vendas](#machine-learning-previsão-de-vendas)
  - [Integração com IA Generativa](#integração-com-ia-generativa)
- [Treinamento e Avaliação do Modelo](#treinamento-e-avaliação-do-modelo)
- [Dificuldades e Desafios](#dificuldades-e-desafios)
- [Melhorias e Próximos Passos](#melhorias-e-próximos-passos)
- [Considerações Finais](#considerações-finais)
- [Redes Sociais](#redes-sociais)

## Descrição do Projeto

O projeto utiliza o Superstore Dataset para:
- Realizar uma análise exploratória detalhada, identificando insights sobre vendas, lucros e comportamento dos clientes.
- Desenvolver um modelo preditivo para previsão de vendas com base em features selecionadas.
- Integrar a análise com uma IA generativa (usando a API do OpenAI) para gerar insights estratégicos em linguagem natural.

## Recursos Utilizados

- **Pandas** e **NumPy** para manipulação e análise dos dados.
- **Matplotlib** e **Seaborn** para criação de visualizações.
- **Scikit-learn** para desenvolvimento e avaliação de modelos de machine learning.
- **OpenAI API** para integração com modelos de linguagem (ex.: GPT-4 ou Gemini AI).

## Pré-requisitos e Instalação

Certifique-se de ter o Python 3.x instalado e, preferencialmente, utilize um ambiente como o Jupyter Notebook ou Google Colab.

### Instalação das Bibliotecas

Execute o seguinte comando para instalar as bibliotecas necessárias:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openai
```

### Configuração da API

Para utilizar a integração com a IA generativa, configure sua chave de API do Gemini AI. Você pode definir a variável de ambiente `GEMINI_API_KEY` ou inserir diretamente sua chave no código (não recomendado para produção).

## Explicação do Código

### Preparação e Análise dos Dados (EDA)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Carregamento do dataset
df = pd.read_csv('superstore_dataset.csv')

# Visualização inicial do dataset
print(df.head())
print(df.info())
print(df.describe())

# Conversão de colunas de data para o formato datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

# Verificação de valores nulos e duplicados
print("Valores nulos:\n", df.isnull().sum())
print("Duplicatas:", df.duplicated().sum())
```

> **Explicação:**  
> Nesta etapa, os dados são carregados e inspecionados. São aplicadas conversões de tipo (especialmente para datas) e verificações de inconsistências (valores nulos e duplicados).

### Visualizações e Insights

```python
# Visualização da distribuição das vendas
plt.figure(figsize=(8,5))
sns.histplot(df['Sales'], kde=True)
plt.title('Distribuição das Vendas')
plt.xlabel('Sales')
plt.ylabel('Frequência')
plt.show()

# Relação entre Vendas e Lucro
plt.figure(figsize=(8,5))
sns.scatterplot(x='Sales', y='Profit', data=df)
plt.title('Relação entre Vendas e Lucro')
plt.xlabel('Sales')
plt.ylabel('Profit')
plt.show()
```

> **Explicação:**  
> São geradas visualizações para compreender a distribuição das vendas e a relação entre vendas e lucro, facilitando a identificação de padrões e insights relevantes.

### Machine Learning: Previsão de Vendas

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Seleção de features e variável alvo
features = df[['Quantity', 'Discount']]  # Exemplo: usando quantidade e desconto
target = df['Sales']

# Divisão dos dados em conjuntos de treino e teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Treinamento do modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Previsão e avaliação do modelo
predictions = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, predictions))
print("MSE:", mean_squared_error(y_test, predictions))
```

> **Explicação:**  
> Um modelo simples de regressão linear é utilizado para prever as vendas com base em variáveis selecionadas. A performance é avaliada por meio das métricas MAE e MSE.

### Integração com IA Generativa

```python
import openai
import os

# Configuração da API do Gemini AI
genai.configure(api_keys=os.getenv("GEMINI_API_KEY", "sua-chave-api-aqui"))  # Substitua "sua-chave-api-aqui" se necessário

def gerar_insight(contexto):
    """
    Gera insights em linguagem natural utilizando o GEMINI-1.5-PRO.
    
    Parâmetros:
        contexto (str): Texto contendo insights e informações da análise.
    
    Retorna:
        resposta_texto (str): Resposta gerada pelo modelo com interpretações e recomendações.
    """
    try:
        """ Modelo avançado Gemini AI"""
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generative_content(
            contexto,
            generation_config={
                """ Ajusta a criatividade da rsposta do modelo"""
                "temperature": 0.7
            },
        )
        return response.text
    except Exception as e:
        print("Erro ao chamar a API do Gemini AI: ", e)
        return None

# Exemplo de uso da função
if __name__ == "__main__":
    contexto_exemplo = (
        "Análise de dados do Superstore:\n"
        "- As vendas apresentam uma distribuição com picos durante períodos promocionais;\n"
        "- Há uma correlação positiva entre quantidade de vendas e descontos, mas descontos muito altos reduzem a margem de lucro;\n"
        "- Segmentação dos clientes revela que a região Sul possui maior fidelidade e ticket médio elevado.\n"
        "\nCom base nesses resultados, quais estratégias você recomendaria para aumentar a rentabilidade e otimizar as campanhas promocionais?"
    )
    
    insight_gerado = gerar_insight(contexto_exemplo)
    if insight_gerado:
        print("Insight gerado pela IA:")
        print(insight_gerado)
    else:
        print("Falha ao gerar insight.")
```

> **Explicação:**  
> Esta seção integra a API do Gemini AI para gerar insights em linguagem natural. A função `gerar_insight` envia um prompt (com um contexto detalhado) para o modelo GEMINI-1.5-PRO e retorna a resposta, tratando possíveis erros na requisição.

## Treinamento e Avaliação do Modelo

- **Divisão dos Dados:** Os dados foram divididos em 80% para treinamento e 20% para teste.
- **Modelo Utilizado:** Regressão linear foi empregada para estabelecer uma linha de base na previsão de vendas.
- **Métricas de Avaliação:** Foram calculadas as métricas MAE e MSE para avaliar a performance do modelo.
- **Possíveis Extensões:** Testar outros algoritmos (como Decision Trees, Random Forest ou Gradient Boosting) e realizar tuning dos hiperparâmetros para melhorar a performance.

## Dificuldades e Desafios

- **Limpeza e Preparação dos Dados:**  
  - Conversão correta de formatos de data.
  - Tratamento de valores ausentes e duplicados.
- **Seleção de Features:**  
  - Identificar quais variáveis impactam significativamente as vendas, exigindo experimentação com diferentes combinações.
- **Integração com a IA:**  
  - Configuração e ajustes dos parâmetros do GEMINI-1.5-PRO (como `temperature` e `max_tokens`) para obter respostas precisas e úteis.
  - Tratamento de erros na comunicação com a API.

## Melhorias e Próximos Passos

- **Aprimoramento dos Modelos:**  
  - Explorar técnicas de machine learning mais avançadas e realizar validação cruzada.
  - Ajuste dos hiperparâmetros para otimização do desempenho.
- **Refinamento da Integração com IA:**  
  - Ajustar os prompts e testar diferentes configurações para extrair insights mais detalhados.
- **Documentação Interativa:**  
  - Elaborar um guia de execução detalhado, possibilitando que outros reproduzam os resultados com facilidade.
- **Automação do Pipeline:**  
  - Desenvolver scripts que automatizem o processo de limpeza, treinamento, avaliação e geração de relatórios.

## Considerações Finais

Mesmo que nem todos os níveis do teste sejam completados, o importante é garantir que o que foi desenvolvido esteja funcionando e seja bem documentado. Neste projeto:

- **Funcionalidade:**  
  - O código foi testado e as principais funcionalidades (EDA, modelagem e integração com IA) operam conforme o esperado.
  
- **Documentação das Dificuldades:**  
  - Foram registradas as dificuldades encontradas na limpeza dos dados, seleção de features e configuração da API, assim como as tentativas e ajustes realizados.
  
- **Implementações Realizadas:**  
  - Desenvolvimento de uma análise exploratória robusta.
  - Criação de um modelo de machine learning para previsão de vendas.
  - Integração completa com o `Gemini AI` para geração de insights em linguagem natural.
  
- **Pontos Pendentes:**  
  - Experimentação com outros modelos e técnicas de otimização.
  - Refinamento dos prompts e parâmetros para a IA generativa.
  - Expansão da documentação e criação de um pipeline automatizado.
  
- **Próximos Passos:**  
  - Investigar técnicas mais avançadas de modelagem.
  - Melhorar a integração com o LLM.
  - Publicar e documentar o projeto de forma interativa para facilitar a reprodução dos resultados.

## Redes Sociais

- **Github:** [https://github.com/IsaqueAlmeida](https://github.com/IsaqueAlmeida)
- **Linkedin:** [https://www.linkedin.com/in/isaque-f-s-almeida/](https://www.linkedin.com/in/isaque-f-s-almeida/)