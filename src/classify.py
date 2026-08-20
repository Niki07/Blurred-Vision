"""Send a single CIFAR-10(-C) image to a model via OpenRouter and get a label back."""

import base64
from io import BytesIO

from PIL import Image

from src.config import CLASSIFICATION_PROMPT


def image_array_to_base64(image_array):
    buffer = BytesIO()
    Image.fromarray(image_array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def classify_image(client, model_id, image_array):
    """Return the model's predicted CIFAR-10 label (lowercase, stripped) for one image."""
    base64_image = image_array_to_base64(image_array)

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
    )
    return response.choices[0].message.content.strip().lower()
