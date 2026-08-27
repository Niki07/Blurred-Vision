"""Experiment driver.

Runs one model's full sweep (every severity x the fixed image sample) at a
time, then pauses for manual confirmation before starting the next model —
so results can be reviewed model-by-model instead of waiting for one long
run. See ENGINEERING_DECISIONS.md (2026-08-26) for why.

Resume-safe: any (model, blur_type, severity, image_id) already present in
results.csv is skipped, so an interrupted or manually-paused run can always
be restarted without re-paying for completed calls.

Currently scoped to motion_blur only (see ENGINEERING_DECISIONS.md) — rerun
with BLUR_TYPE/CORRUPTION_FILE changed for glass_blur / defocus_blur later;
existing rows are untouched since they key on blur_type too.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from src.classify import classify_image
from src.client import get_client
from src.config import CONCURRENCY, IMAGES_PER_CLASS, INT_TO_LABEL, MODEL_IDS, SEVERITIES
from src.data import build_fixed_sample, get_image, load_corruption, load_labels
from src.persist import append_result, load_completed_keys

RESULTS_CSV = "results/results.csv"
BLUR_TYPE = "motion_blur"
CORRUPTION_FILE = "data/motion_blur.npy"

# Touch this file to release the pause between models. File-based rather than
# input() so the run can be driven from outside its own terminal (e.g. a
# supervising script/process) as well as interactively.
CONTINUE_SIGNAL_PATH = "results/.continue_signal"


def run_model(client, model_key, model_id, sample, labels, corruption, completed, lock):
    """Classify every (severity, image) pair in `sample` for one model.

    Skips pairs already in `completed`. Runs the remaining calls across a
    small thread pool for speed, writing each result to disk as it lands.
    """
    tasks = [
        (severity, image_id)
        for severity in SEVERITIES
        for image_id in sample
        if (model_key, BLUR_TYPE, severity, image_id) not in completed
    ]

    total = len(sample) * len(SEVERITIES)
    if not tasks:
        print(f"{model_key}: all {total} rows already done, nothing to do.")
        return

    print(f"{model_key}: {len(tasks)}/{total} calls remaining.")

    def do_one(task):
        severity, image_id = task
        image_array = get_image(corruption, image_id, severity)
        true_label = INT_TO_LABEL[int(labels[image_id])]

        prediction = None
        for attempt in range(3):
            try:
                prediction = classify_image(client, model_id, image_array)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  ERROR {model_key} sev{severity} img{image_id}: {e}")
                    return
                time.sleep(2 * (attempt + 1))

        row = {
            "model": model_key,
            "blur_type": BLUR_TYPE,
            "severity": severity,
            "image_id": image_id,
            "true_label": true_label,
            "prediction": prediction,
            "correct": int(prediction == true_label),
        }
        with lock:
            append_result(row, RESULTS_CSV)
            completed.add((model_key, BLUR_TYPE, severity, image_id))

    finished = 0
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        for _ in executor.map(do_one, tasks):
            finished += 1
            if finished % 200 == 0:
                print(f"  {model_key}: {finished}/{len(tasks)}")

    print(f"{model_key}: finished this pass ({len(tasks)} calls attempted).")


def main():
    client = get_client()
    labels = load_labels("data/labels.npy")
    corruption = load_corruption(CORRUPTION_FILE)
    sample = build_fixed_sample(labels, IMAGES_PER_CLASS)
    completed = load_completed_keys(RESULTS_CSV)
    lock = threading.Lock()

    model_items = list(MODEL_IDS.items())
    total_planned = len(sample) * len(SEVERITIES) * len(model_items)
    print(
        f"Blur type: {BLUR_TYPE} | sample: {len(sample)} images | "
        f"severities: {len(SEVERITIES)} | models: {len(model_items)} | "
        f"total calls planned: {total_planned}"
    )

    for i, (model_key, model_id) in enumerate(model_items):
        print(f"\n=== [{i + 1}/{len(model_items)}] {model_key} ({model_id}) ===")
        run_model(client, model_key, model_id, sample, labels, corruption, completed, lock)
        print("done")

        if i < len(model_items) - 1:
            next_model = model_items[i + 1][0]
            print(f"Waiting to continue to {next_model} (touch {CONTINUE_SIGNAL_PATH})...")
            while not os.path.exists(CONTINUE_SIGNAL_PATH):
                time.sleep(1)
            os.remove(CONTINUE_SIGNAL_PATH)
            print(f"Continuing to {next_model}...")


if __name__ == "__main__":
    main()
