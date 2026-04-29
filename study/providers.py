import os
from dataclasses import dataclass
from typing import Protocol

import requests


class Provider(Protocol):
    provider_name: str

    def is_available(self) -> bool: ...

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str: ...


@dataclass
class OpenAIProvider:
    model: str
    provider_name: str = "openai"

    def __post_init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "input": prompt,
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data["output"][0]["content"][0]["text"]


@dataclass
class GeminiProvider:
    model: str
    provider_name: str = "gemini"

    def __post_init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


@dataclass
class GroqProvider:
    model: str
    provider_name: str = "groq"

    def __post_init__(self) -> None:
        self.api_key = os.getenv("GROQ_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
