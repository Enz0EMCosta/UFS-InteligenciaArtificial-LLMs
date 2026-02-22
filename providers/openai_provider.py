import os
import openai
from typing import List, Dict
from core.interfaces import LLMProvider
from core.config import ModelConfig

class OpenAIProvider(LLMProvider):
    """Implementação concreta para a API da OpenAI."""
    
    def __init__(self):
        # A chave DEVE vir de variável de ambiente (.env ou Hugging Face Secrets)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("A variável de ambiente 'OPENAI_API_KEY' não está configurada.")
        self.client = openai.OpenAI(api_key=api_key)

    def call_api(self, config: ModelConfig, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=config.model_name,
            messages=messages,
            temperature=config.temperature
        )
        return response.choices[0].message.content