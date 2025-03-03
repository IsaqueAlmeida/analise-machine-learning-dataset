""" Integrando com a IA Generativa """

# Gerando insights com a API do Gemini AI
# Importando biblioteca da Gemini AI
# import teste_machine_learning as tml
import google.generativeai as genai
import os

# Configurando a chave da API
genai.configure(api_key=os.getenv('GEMINI_API_KEY',
                                  'Sua_Chave_Key_Gemini_AI'))

# Criando a função para gerar insights


def gerar_insights(contexto):
    """
      Servirá para gerar insights em linguagem natural utilizando um modelo
      de linguagem (LLM) do Gemini AI.

      Parâmetros:
        - contexto (str): Texto contendo informações e insights derivados do
          EDA ou da modelagem.

      Retorna:
        - resposta_texto (str): Resposta gerada pelo modelo, contendo uma
          interpretação e recomendações.
    """
    try:
        # Modelo avançado Gemini AI
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(
            contexto,
            generation_config={
                # Ajusta a criatividade da resposta do modelo
                "temperature": 0.7
            },
        )
        return response.text
    except Exception as e:
        print('Erro ao chamar a API do Gemini AI: ', e)
        return None

# Uso da função


if __name__ == '__main__':
    # Exemplo de contexto com insights derivados da análise de dados
    contexto_exemplo = (

        "Análise de dados do Superstore:\n"
        "- Quero cagar - será que papel higiênico resolve??\n"
        "- Faça uma análise dos gráficos já gerados - faça um resumo;\n"
        "- Avalie a relação entre as vendas e os lucros;\n"
        "- Identifique padrões e tendências nos dados;\n"
        """- Há uma correlação positiva entre quantidade de vendas e
           descontos, mas descontos muito altos reduzem a margem de lucro;\n"""
        "- Sugira ações para melhorar a lucratividade das vendas;\n"
        """- Segmentação dos clientes revela que a região Sul possui maior
             fidelidade ticket médio elevado.\n
        """
        """- "\nCom base nesses resultados, quais estratégias você
               recomendaria para aumentara rentabilidade e otimizar as
               campanhas promocionais?
        """
    )

    insight_gerado = gerar_insights(contexto_exemplo)
    if insight_gerado:
        print('Insight gerado pela IA: ')
        print(insight_gerado)
    else:
        print('Falha ao gerar insight!')
