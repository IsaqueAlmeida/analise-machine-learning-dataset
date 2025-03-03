""" Integrando com a IA Generativa """

# Gerando insights com a API do Gemini AI
# Importando biblioteca da Gemini AI
# import teste_machine_learning as tml
import google.generativeai as genai
import os

# Configurando a chave da API
genai.configure(api_key=os.getenv('GEMINI_API_KEY',
                                  'Sua_chave_Key_Gemini_AI'))

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
    while True:
        user_prompt = input(
            "Digite a sua pergunta a Gemini (ou 'sair' para encerrar): "
          )
        if user_prompt.lower() == 'sair':
            break
        insight_gerado = gerar_insights(user_prompt)
        if insight_gerado:
            print("\nIA respondendo: ")
            print(insight_gerado)
        else:
            print("\nFalha ao gerar insights com a Gemini AI!")
