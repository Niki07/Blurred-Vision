"""OpenRouter client setup."""

import os

from dotenv import load_dotenv
from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_client():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Copy .env.example to .env and fill in your key."
        )
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
