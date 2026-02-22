from typing import List, Dict, Callable
from core.config import ModelConfig
from core.interfaces import LLMProvider

class ContextManager:
    """Responsável pela formatação de mensagens e manutenção segura da memória."""
    
    @staticmethod
    def create_message(role: str, content: str) -> Dict[str, str]:
        return {"role": role, "content": content}

    @staticmethod
    def truncate_history(history: List[Dict[str, str]], limit: int, token_estimator: Callable[[str], int]) -> List[Dict[str, str]]:
        """
        Garante que a lista de mensagens não exceda o limite de tokens da API.
        Preserva a instrução de sistema (role='system') sempre no topo.
        """
        if not history:
            return []

        system_prompt = []
        conversation = history[:]

        # Isola o System Prompt para que nunca seja apagado
        if conversation and conversation[0].get("role") == "system":
            system_prompt = [conversation.pop(0)]

        # Remove as mensagens mais antigas até que o total caiba no limite
        while True:
            total_tokens = sum(token_estimator(m["content"]) for m in system_prompt + conversation)
            if total_tokens <= limit or not conversation:
                break
            # Remove a mensagem de índice 0 (a mais antiga da interação)
            conversation.pop(0)

        return system_prompt + conversation

class ConversationalAgent:
    """Orquestrador principal que liga a entrada do usuário à API, mantendo a memória."""
    
    def __init__(self, config: ModelConfig, provider: LLMProvider):
        self.config = config
        self.provider = provider
        self.history: List[Dict[str, str]] = []
        # Estimativa simples (1 token ~= 4 caracteres). Pode ser substituído por bibliotecas como tiktoken.
        self.token_estimator = lambda text: len(text) // 4

    def set_system_prompt(self, prompt: str):
        """Define ou atualiza a instrução de comportamento do agente."""
        msg = ContextManager.create_message("system", prompt)
        if self.history and self.history[0]["role"] == "system":
            self.history[0] = msg
        else:
            self.history.insert(0, msg)

    def chat(self, user_input: str) -> str:
        """Fluxo de execução: Mensagem -> Truncamento -> API -> Resposta -> Memória"""
        # 1. Constrói a mensagem do usuário
        current_msg = ContextManager.create_message("user", user_input)
        self.history.append(current_msg)

        # 2. Faz a Gestão de Contexto
        payload = ContextManager.truncate_history(
            history=self.history,
            limit=self.config.max_tokens,
            token_estimator=self.token_estimator
        )

        # 3. Chama o provedor (Agnóstico à implementação técnica)
        reply_text = self.provider.call_api(self.config, payload)

        # 4. Salva a resposta do assistente na memória
        reply_msg = ContextManager.create_message("assistant", reply_text)
        self.history.append(reply_msg)

        return reply_text