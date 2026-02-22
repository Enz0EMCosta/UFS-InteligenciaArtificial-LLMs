import gradio as gr
import os
from dotenv import load_dotenv # 1. Importe a biblioteca
load_dotenv() # Carrega as variáveis de ambiente do arquivo .env que contém as chaves de API
from core.config import ModelConfig
from core.agent import ConversationalAgent
from providers.gemini_provider import GeminiProvider #importando a subclasse da gemini exclusivamente para teste 


# ESCOLHA O SEU PROVEDOR AQUI
# Para trocar para o Gemini, comente a linha da OpenAI e descomente a do Gemini
from providers.openai_provider import OpenAIProvider
# from providers.gemini_provider import GeminiProvider

# 1. Configuração do Modelo
config = ModelConfig(
    provider="openai",
    model_name="gemini-3-flash", # Ou "gemini-3-flash" se usar o Google
    temperature=0.7,
    max_tokens=2000 # Limite de memória antes do truncamento
)

# 2. Inicialização do Orquestrador
# Nota: Lembre-se de configurar a variável de ambiente PROVEDOR_API_KEY antes de rodar!
try:
    #provider = OpenAIProvider()
    provider = GeminiProvider() 
    
    agente = ConversationalAgent(config, provider)
    agente.set_system_prompt("Você é um assistente útil. Responda de forma clara e objetiva.")
except Exception as e:
    print(f"Erro de inicialização: {e}")
    agente = None

# 3. Função de Ligação (Interface <-> Core)
def interagir(mensagem: str, historico: list) -> str:
    if not agente:
        return "Erro: O agente não foi inicializado corretamente (verifique suas chaves de API)."
    try:
        return agente.chat(mensagem)
    except Exception as e:
        return f"Ocorreu um erro na chamada da API: {str(e)}"

# 4. Interface Gradio (Ideal para Hugging Face Spaces)
demo = gr.ChatInterface(
    fn=interagir,
    title="🤖 Framework Multimodelo Agnóstico",
    description="Implementação de Orquestração, Engenharia de Prompt e Gestão de Memória (Truncamento).",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()