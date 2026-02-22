import os
from google import genai
from google.genai import types
from typing import List, Dict
from core.interfaces import LLMProvider
from core.config import ModelConfig

class GeminiProvider(LLMProvider):
    """Implementação concreta para a API do Google Gemini (Novo SDK)."""
    
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("A variável de ambiente 'GEMINI_API_KEY' não está configurada.")
        # O novo SDK usa genai.Client()
        self.client = genai.Client(api_key=api_key)

    def call_api(self, config: ModelConfig, messages: List[Dict[str, str]]) -> str:
        formatted_messages = []
        
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            content = f"Instrução de Sistema: {msg['content']}" if msg["role"] == "system" else msg["content"]
            
            # Formatação exigida pelo novo SDK google.genai
            formatted_messages.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=content)]
                )
            )
            
        response = self.client.models.generate_content(
            model=config.model_name,
            contents=formatted_messages,
            config=types.GenerateContentConfig(
                temperature=config.temperature,
            )
        )
        return response.text