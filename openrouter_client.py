import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "qwen/qwen3.5-9b",
)

if not API_KEY:
    raise RuntimeError(
        "No se encontró OPENROUTER_API_KEY en el archivo .env"
    )

client = OpenAI(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
    timeout=90.0,
    max_retries=3,
)


def call_qwen(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 200,
) -> tuple[str, dict[str, Any]]:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("El modelo devolvió una respuesta vacía.")

    usage = response.usage

    metadata = {
        "model": response.model,
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0,
    }

    return content, metadata