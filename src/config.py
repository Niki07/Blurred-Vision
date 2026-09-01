"""Fixed constants for the blur-robustness experiment."""

# Fixed 9-model roster (see ENGINEERING_DECISIONS.md for rationale/dates).
# openrouter/free must NOT be used for real data — it silently swaps the underlying
# model, which breaks reproducibility and the "record exact model versions" requirement.
MODEL_IDS = {
    "claude_opus": "anthropic/claude-opus-5",
    "claude_sonnet": "anthropic/claude-sonnet-5",
    "claude_haiku": "anthropic/claude-haiku-4.5",
    "gpt_sol": "openai/gpt-5.6-sol",
    "gpt_terra": "openai/gpt-5.6-terra",
    "gpt_luna": "openai/gpt-5.6-luna",
    "gemini": "google/gemini-3.1-flash-lite",
    "grok": "x-ai/grok-4.3",
    "kimi": "moonshotai/kimi-k3",
}

# Disables hidden reasoning/thinking tokens — OpenRouter's documented unified
# `reasoning` param (https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).
# Used by default for every call (see get_extra_body() below) because the smoke
# test on 2026-08-26 showed several models spending 100-500+ hidden completion
# tokens "thinking" about a one-word classification task, inflating real cost
# ~3.7x above estimates that assumed a ~2-token answer.
REASONING_OFF_EXTRA_BODY = {"reasoning": {"effort": "none"}}

# Claude Opus 5 needs its own override: reasoning:"none" fully disables its
# thinking, but per known Opus 5 behavior that makes it leak a full paragraph
# of reasoning into the *visible* answer instead of a bare label (confirmed via
# smoke test on 2026-08-26 — completion_tokens=260, a multi-sentence answer,
# where every other model returned 1-8 tokens). Low effort keeps thinking on
# (avoiding the leak) while still cutting cost well below the default.
CLAUDE_OPUS_EXTRA_BODY = {"reasoning": {"effort": "low"}}


def get_extra_body(model_id):
    """Per-model default extra_body for classify_image — see comments above."""
    if model_id == MODEL_IDS["claude_opus"]:
        return CLAUDE_OPUS_EXTRA_BODY
    return REASONING_OFF_EXTRA_BODY

# CONFIRMED NON-FUNCTIONAL via smoke test on 2026-08-26: prompt_tokens came back
# ~1134 (the default/high-res cost) with this field set, not the ~320 expected
# for low-res — OpenRouter does not appear to forward this field to Gemini via
# the OpenAI-compatible endpoint. Not wired into classify.py. Left here in case
# a different field name/path is found later.
GEMINI_LOW_RES_EXTRA_BODY_UNUSED = {"generation_config": {"media_resolution": "low"}}

BLUR_TYPES = ["motion_blur", "glass_blur", "defocus_blur"]

SEVERITIES = [1, 2, 3, 4, 5]

IMAGES_PER_LEVEL = 10000

CIFAR10_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

LABEL_TO_INT = {label: i for i, label in enumerate(CIFAR10_LABELS)}
INT_TO_LABEL = {i: label for i, label in enumerate(CIFAR10_LABELS)}

CLASSIFICATION_PROMPT = """
Classify this CIFAR-10 image.

Choose exactly ONE label:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Respond with only the label.
"""

# 0 (not each provider's default, typically 1.0) so every model gives its
# single most-confident answer instead of a randomly sampled one — removes
# run-to-run sampling noise as a confound on top of the blur-severity effect
# being measured. See ENGINEERING_DECISIONS.md, 2026-08-28.
CLASSIFICATION_TEMPERATURE = 0

RESULT_FIELDS = [
    "model",
    "blur_type",
    "severity",
    "image_id",
    "true_label",
    "prediction",
    "correct",
]

# Richer schema used by the real data-collection pipeline (src/pipeline.py)
# and the pilot script — adds exact model ID, the model's raw (non-normalized)
# text, whether that text was a valid CIFAR-10 label, request latency, and
# OpenRouter's reported per-call cost. See ENGINEERING_DECISIONS.md.
DETAILED_RESULT_FIELDS = [
    "model",
    "model_id",
    "blur_type",
    "severity",
    "image_id",
    "true_label",
    "raw_response",
    "prediction",
    "valid_response",
    "correct",
    "latency_seconds",
    "cost_usd",
]
