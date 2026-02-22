from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable

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