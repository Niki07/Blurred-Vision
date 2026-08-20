"""Fixed constants for the blur-robustness experiment."""

# TODO: replace with exact OpenRouter model slugs before running the real experiment.
# openrouter/free must NOT be used for final data — it silently swaps the underlying
# model, which breaks reproducibility and the "record exact model versions" requirement.
MODEL_IDS = {
    "gpt": "TODO",
    "claude": "TODO",
    "gemini": "TODO",
}

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

RESULT_FIELDS = [
    "model",
    "blur_type",
    "severity",
    "image_id",
    "true_label",
    "prediction",
    "correct",
]
