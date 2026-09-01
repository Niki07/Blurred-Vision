"""OpenRouter client setup."""

import os

import httpx2
from dotenv import load_dotenv
from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _get_api_key():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Copy .env.example to .env and fill in your key."
        )
    return api_key


def get_client():
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=_get_api_key())


def get_remaining_credits():
    """Real remaining OpenRouter balance (total_credits - total_usage), in USD.

    Hits OpenRouter's own accounting endpoint directly (no API call to a
    model), so this reflects actual spend rather than an estimate.
    """
    response = httpx2.get(
        f"{OPENROUTER_BASE_URL}/credits",
        headers={"Authorization": f"Bearer {_get_api_key()}"},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()["data"]
    return data["total_credits"] - data["total_usage"]
