from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable
import google.generativeai as genai # Requer instalação da biblioteca para o gemini
import openai # Requer instalação da biblioteca (mesmo com openai)
# outras ias podem requerer outras bibliotecas apenas openai e gemini vão ser usadas no teste 


'''o uso do codigo implica a criação de uma subclasse herdada baseada 
no modelo qu'''

# ==========================================
# 1. Estrutura de Configuração (Base)
# ==========================================
@dataclass
class ModelConfig:
    """
    Parâmetros universais de configuração para qualquer modelo.
    """
    provider: str
    model_name: str
    temperature: float
    max_tokens: int
    modality: str = "text"


# ==========================================
# 2. Contratos e Interfaces (Generalização)
# ==========================================
class LLMProvider(ABC):
    """
    Interface base para qualquer provedor de IA.
    Para adicionar uma nova IA (Kimi, Gemini, OpenAI, Claude), 
    basta criar uma classe que herde desta e implementar o método `call_api`.
    """
    
    @abstractmethod
    def call_api(self, config: ModelConfig, messages: List[Dict[str, str]]) -> str:
        """
        Recebe as mensagens padronizadas e a configuração, faz a requisição HTTP/SDK
        para a API específica e retorna apenas o texto gerado (Pós-processamento).
        """
        pass


# ==========================================
# 3. Gestão de Contexto e Memória
# ==========================================
class ContextManager:
    """
    Responsável por formatar mensagens e garantir que o limite de tokens não seja excedido.
    """
    
    @staticmethod
    def create_message(role: str, content: str) -> Dict[str, str]:
        return {"role": role, "content": content}

    @staticmethod
    def truncate_history(history: List[Dict[str, str]], limit: int, token_estimator: Callable[[str], int]) -> List[Dict[str, str]]:
        """
        Garante a manutenção da memória sem estourar o limite da API.
        Preserva a instrução de sistema (role="system") sempre no topo.
        """
        if not history:
            return []

        system_prompt = []
        conversation = history[:]

        # Isola o System Prompt para proteção
        if conversation[0].get("role") == "system":
            system_prompt = [conversation.pop(0)]

        # Remove as mensagens mais antigas (índice 0 da conversation) até caber no limite
        while True:
            total_tokens = sum(token_estimator(m["content"]) for m in system_prompt + conversation)
            if total_tokens <= limit or not conversation:
                break
            conversation.pop(0)

        return system_prompt + conversation


# ==========================================
# 4. Agente Conversacional (Orquestrador)
# ==========================================
class ConversationalAgent:
    """
    Orquestra a entrada do usuário, a gestão do contexto e a chamada ao provedor LLM.
    """
    def __init__(self, config: ModelConfig, provider: LLMProvider):
        self.config = config
        self.provider = provider
        self.history: List[Dict[str, str]] = []
        
        # Função simples de estimativa de tokens (1 token ~= 4 caracteres). 
        # Pode ser sobrescrita por tokenizadores reais (ex: tiktoken).
        self.token_estimator = lambda text: len(text) // 4

    def set_system_prompt(self, prompt: str):
        """Define o comportamento inicial da IA."""
        msg = ContextManager.create_message("system", prompt)
        # Garante que o system prompt seja sempre o primeiro
        if self.history and self.history[0]["role"] == "system":
            self.history[0] = msg
        else:
            self.history.insert(0, msg)

    def chat(self, user_input: str) -> str:
        """
        Fluxo principal do pseudocódigo:
        1. Cria mensagem -> 2. Trunca Histórico -> 3. Chama API -> 4. Salva Resposta
        """
        # 1. Construção da Mensagem
        current_msg = ContextManager.create_message("user", user_input)
        self.history.append(current_msg)

        # 2. Gestão de Contexto
        payload = ContextManager.truncate_history(
            history=self.history,
            limit=self.config.max_tokens,
            token_estimator=self.token_estimator
        )

        # 3 e 4. Chamada de Inferência e Extração (encapsulado no Provider)
        reply_text = self.provider.call_api(self.config, payload)

        # Salva a resposta no histórico para a próxima iteração
        reply_msg = ContextManager.create_message("assistant", reply_text)
        self.history.append(reply_msg)

        return reply_text

# ==========================================
# 5. IMPLEMENTAÇÕES DOS PROVEDORES (Classes Herdadas entram AQUI!)
# ==========================================

class OpenAIProvider(LLMProvider):
    def call_api(self, config: ModelConfig, messages: List[Dict[str, str]]) -> str:
        # Instanciar cliente (A chave da API deve vir de variáveis de ambiente .env)
        client = openai.OpenAI() 
        
        response = client.chat.completions.create(
            model=config.model_name,
            messages=messages,
            temperature=config.temperature
        )
        return response.choices[0].message.content



class GeminiProvider(LLMProvider):
    def call_api(self, config: ModelConfig, messages: List[Dict[str, str]]) -> str:
        # O Gemini usa nomenclaturas diferentes (ex: "model" em vez de "assistant")
        # O provedor tem a responsabilidade de traduzir o padrão do framework para a API
        formatted_messages = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else msg["role"]
            formatted_messages.append({"role": role, "parts": [msg["content"]]})
            
        model = genai.GenerativeModel(config.model_name)
        response = model.generate_content(
            formatted_messages,
            generation_config={"temperature": config.temperature}
        )
        return response.text


# 1. Escolha a configuração e o provedor
config = ModelConfig(
    provider="openai",
    model_name="gpt-4o", #aparentemente gpt4o nn existe mais
    temperature=0.7,
    max_tokens=2000 # Limite de memória do contexto
)
provider = OpenAIProvider() # ou GeminiProvider(), KimiProvider()...

# 2. Instancie o Agente 
agent = ConversationalAgent(config, provider)
agent.set_system_prompt("Você é um assistente especialista em programação. Responda de forma concisa.")

# 3. Utilize em um loop, endpoint de API, ou interface gráfica
resposta = agent.chat("Olá, como você gerencia sua memória?")
print(resposta)

resposta_2 = agent.chat("Quais foram as exatas palavras da minha pergunta anterior?")
print(resposta_2)