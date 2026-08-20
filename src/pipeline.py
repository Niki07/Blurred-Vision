"""Experiment driver.

STATUS: proof-of-concept only, ported from the original Colab notebook
(notebooks/proof_of_concept.py). This currently reproduces that single-image,
single-model, single-blur-type test using the persistence-safe append pattern.

TODO before this is the real experiment:
  - fill in src/config.py MODEL_IDS with fixed, explicitly named OpenRouter model IDs
  - decide on the fixed image sample (which base image indices, how many per class)
  - loop over MODEL_IDS x BLUR_TYPES x SEVERITIES x chosen image sample
  - point RESULTS_CSV at a persistent location (e.g. a mounted Google Drive path)
    if running in Colab, since local runtime storage can be lost on disconnect
"""

from src.classify import classify_image
from src.client import get_client
from src.config import INT_TO_LABEL, LABEL_TO_INT
from src.data import get_image, indices_for_label, load_corruption, load_labels

RESULTS_CSV = "results/results.csv"


def main():
    from src.persist import append_result

    client = get_client()

    labels = load_labels("data/labels.npy")
    defocus = load_corruption("data/defocus_blur.npy")

    cat_indices = indices_for_label(labels, LABEL_TO_INT["cat"])
    base_image_index = int(cat_indices[0])
    severity = 1

    image_array = get_image(defocus, base_image_index, severity)

    # TODO: swap "openrouter/free" for a real MODEL_IDS entry once fixed model
    # IDs are chosen — "free" must not be used for final data (see config.py).
    prediction = classify_image(client, "openrouter/free", image_array)
    true_label = INT_TO_LABEL[LABEL_TO_INT["cat"]]

    row = {
        "model": "openrouter_free_test",
        "blur_type": "defocus",
        "severity": severity,
        "image_id": base_image_index,
        "true_label": true_label,
        "prediction": prediction,
        "correct": int(prediction == true_label),
    }
    append_result(row, RESULTS_CSV)
    print(row)


if __name__ == "__main__":
    main()
