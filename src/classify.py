"""Send a single CIFAR-10(-C) image to a model via OpenRouter and get a label back."""

import base64
import time
from io import BytesIO

from PIL import Image

from src.config import CLASSIFICATION_PROMPT, CLASSIFICATION_TEMPERATURE, get_extra_body


def image_array_to_base64(image_array):
    buffer = BytesIO()
    Image.fromarray(image_array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def classify_image(client, model_id, image_array, extra_body=None):
    """Return the model's predicted CIFAR-10 label (lowercase, stripped) for one image.

    extra_body: dict of provider-specific request fields passed through to
    OpenRouter as-is. Defaults to get_extra_body(model_id) (see config.py) —
    pass an explicit {} to leave reasoning at the model's own default.
    """
    if extra_body is None:
        extra_body = get_extra_body(model_id)
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
        temperature=CLASSIFICATION_TEMPERATURE,
        extra_body=extra_body,
    )
    return response.choices[0].message.content.strip().lower()


def classify_image_with_metrics(client, model_id, image_array, extra_body=None):
    """Like classify_image, but also returns the raw (non-normalized) response
    text, request latency, and OpenRouter's reported cost for the call.

    Needed for research-analysis columns (raw_response, latency_seconds,
    cost_usd, valid_response) that classify_image's simple return value can't
    carry. Requests OpenRouter's cost-accounting field (`usage.cost`) via
    extra_body, confirmed working 2026-08-26.

    Returns (raw_response, prediction, latency_seconds, cost_usd_or_None).
    """
    if extra_body is None:
        extra_body = get_extra_body(model_id)
    extra_body = dict(extra_body)
    extra_body["usage"] = {"include": True}
    base64_image = image_array_to_base64(image_array)

    start = time.monotonic()
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
        temperature=CLASSIFICATION_TEMPERATURE,
        extra_body=extra_body,
    )
    latency_seconds = time.monotonic() - start

    raw_response = response.choices[0].message.content
    prediction = raw_response.strip().lower()
    cost_usd = getattr(response.usage, "cost", None)
    return raw_response, prediction, latency_seconds, cost_usd
