from dataclasses import dataclass

@dataclass
class ModelConfig:
    """
    Parâmetros universais de configuração.
    Independente de qual IA seja usada, estes parâmetros guiarão o comportamento.
    """
    provider: str
    model_name: str
    temperature: float
    max_tokens: int
    modality: str = "text"