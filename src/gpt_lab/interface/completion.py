
import os
from openai import OpenAI
import load_dotenv
from gpt_lab.interface.config import ConfigChat
from typing import List, Dict

load_dotenv.load_dotenv()

class ModelCompletion:
    def __init__(
            self, 
            config: ConfigChat
        ):
        self.model_name = config.model_name
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.stream = config.stream
        # self.max_new_tokens = config.max_new_tokens
        self.remote_completion = config.remote_completion
        self.reasoning_effort = config.reasoning_effort 
        if self.remote_completion:
            self.client = OpenAI(
                base_url=os.environ["OPENAI_BASE_URL"],
                api_key=os.environ["HF_TOKEN"],
            )
        print(f"Model client: {self.client}" if self.remote_completion else "Local model client initialized.")


    def predict(self, message: str, history: List[Dict[str, str]]):
        if not self.remote_completion:
            raise NotImplementedError("Local model prediction not implemented.")

        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=history + [{"role": "user", "content": message}],
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
            stream=self.stream,
            reasoning_effort=self.reasoning_effort,
        )
        accumulated = ""
        accumulated_reasoning = "" 
        is_thinking = False
        for chunk in stream:
            if not chunk.choices:
                break
            if len(chunk.choices) == 0:
                break
            delta = chunk.choices[0].delta

            if is_thinking or (hasattr(delta, "reasoning_content") and delta.reasoning_content):
                if accumulated_reasoning is None:
                    accumulated_reasoning = ""
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    accumulated_reasoning += delta.reasoning_content
                else:
                    accumulated_reasoning += delta.content

            elif hasattr(delta, "content") and delta.content:
                accumulated += delta.content

            if "<think>" in accumulated:
                is_thinking = True
                accumulated_reasoning = accumulated.replace("<think>", "")
                accumulated = ""
            
            if "</think>" in accumulated_reasoning:
                is_thinking = False
                accumulated_reasoning, accumulated = accumulated_reasoning.split("</think>", 1)

            yield {
                "content": accumulated,
                "reasoning": accumulated_reasoning,
                "finish_reason": chunk.choices[0].finish_reason == "stop"
            }
        yield {
            "content": accumulated,
            "reasoning": accumulated_reasoning,
            "finish_reason": True
        }
        
