from abc import ABC, abstractmethod
from typing import List, Dict
from core.config import ModelConfig

class LLMProvider(ABC):
    """
    Interface base para provedores de LLM (OpenAI, Gemini, Kimi, etc).
    Qualquer nova integração deve herdar desta classe e implementar 'call_api'.
    """
    
    @abstractmethod
    def call_api(self, config: ModelConfig, messages: List[Dict[str, str]]) -> str:
        """
        Realiza a chamada HTTP/SDK para a API específica.
        
        Args:
            config: Configurações do modelo (temperatura, max_tokens, etc).
            messages: Lista de dicionários no formato [{"role": "...", "content": "..."}].
            
        Returns:
            str: O texto gerado como resposta pelo modelo.
        """
        pass