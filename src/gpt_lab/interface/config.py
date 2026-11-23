from pydantic import BaseModel

class BaseConfig(BaseModel):
    temperature: float = 1.0
    max_tokens: int = 64
    remote_completion: bool = False
    
class ConfigChat(BaseConfig):
    model_name: str = "gpt-3.5-turbo"
    seed: int = 42
    stream: bool = True
    reasoning_effort: str = "high"

class ConfigCompletion(BaseConfig):
    pass


