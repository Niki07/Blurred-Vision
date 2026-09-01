"""One real call per model: confirms slugs resolve, vision works, and reports
actual token usage — including whether GEMINI_LOW_RES_EXTRA_BODY has any effect.

Uses a synthetic 32x32 image (not real CIFAR-10-C data) since this only checks
API mechanics/cost, not classification accuracy. Costs a few cents to run.

Usage: python -m scripts.smoke_test
"""

import numpy as np

from src.classify import CLASSIFICATION_PROMPT, image_array_to_base64
from src.client import get_client
from src.config import MODEL_IDS, get_extra_body


def call_once(client, model_id, base64_image, extra_body=None):
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CLASSIFICATION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ],
        extra_body=extra_body,
    )
    return response


def main():
    client = get_client()
    rng = np.random.default_rng(0)
    image_array = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    base64_image = image_array_to_base64(image_array)

    for name, model_id in MODEL_IDS.items():
        try:
            response = call_once(client, model_id, base64_image, get_extra_body(model_id))
            prediction = response.choices[0].message.content.strip()
            usage = response.usage
            print(
                f"{name:15s} {model_id:35s} "
                f"prompt_tokens={usage.prompt_tokens:<6} "
                f"completion_tokens={usage.completion_tokens:<4} "
                f"prediction={prediction!r}"
            )
        except Exception as e:
            print(f"{name:15s} {model_id:35s} FAILED: {e}")


if __name__ == "__main__":
    main()
