import os
import google.generativeai as genai
from typing import List, Dict
from core.interfaces import LLMProvider
from core.config import ModelConfig

class GeminiProvider(LLMProvider):
    """Implementação concreta para a API do Google Gemini."""
    
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("A variável de ambiente 'GEMINI_API_KEY' não está configurada.")
        genai.configure(api_key=api_key)

    def call_api(self, config: ModelConfig, messages: List[Dict[str, str]]) -> str:
        # O Gemini tem regras diferentes para a nomenclatura das roles
        formatted_messages = []
        for msg in messages:
            # Ignoramos o system prompt na lista de mensagens padrão do Gemini (deve ser passado na inicialização, 
            # mas para simplificar o wrapper unificado, injetamos como instrução de usuário inicial)
            role = "model" if msg["role"] == "assistant" else "user"
            content = f"Instrução de Sistema: {msg['content']}" if msg["role"] == "system" else msg["content"]
            formatted_messages.append({"role": role, "parts": [content]})
            
        model = genai.GenerativeModel(config.model_name)
        response = model.generate_content(
            formatted_messages,
            generation_config={"temperature": config.temperature}
        )
        return response.text