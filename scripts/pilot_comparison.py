"""Small 9-model pilot: the same 5 image IDs x 5 severities for every model
(25 calls/model, 225 total), instead of the full 140/class run.

Reuses Claude Opus's existing results for those 5 images from results.csv
(no re-calling Opus) and writes everything into a separate pilot CSV —
results/results.csv is never touched. One "model" column (long format),
not one column per model.

Resume-safe: skips any (model, blur_type, severity, image_id) already in
the pilot CSV, so it can be stopped and restarted without re-paying for
completed calls. Each completed row is appended to disk immediately.

Usage: python -m scripts.pilot_comparison
"""

import csv
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from src.client import get_client
from src.classify import CLASSIFICATION_PROMPT, image_array_to_base64
from src.config import CIFAR10_LABELS, MODEL_IDS, SEVERITIES, get_extra_body
from src.data import get_image, load_corruption, load_labels
from src.persist import append_result, load_completed_keys

MAIN_CSV = "results/results.csv"
PILOT_CSV = "results/results_mini.csv"
BLUR_TYPE = "motion_blur"
CORRUPTION_FILE = "data/motion_blur.npy"
ALREADY_RUN_MODEL = "claude_opus"
PILOT_IMAGE_COUNT = 5
CONCURRENCY = 16

PILOT_FIELDS = [
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


def pick_pilot_image_ids():
    """The first 5 distinct image IDs already present for Opus/motion_blur."""
    image_ids = set()
    with open(MAIN_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] == ALREADY_RUN_MODEL and row["blur_type"] == BLUR_TYPE:
                image_ids.add(int(row["image_id"]))
    return sorted(image_ids)[:PILOT_IMAGE_COUNT]


def seed_opus_rows(image_ids, lock, completed):
    """Copy Opus's 25 existing rows for image_ids into the pilot CSV.

    Cost/latency/raw response were never recorded for the original run, so
    those are marked N/A rather than reconstructed or guessed. Resume-aware:
    skips any row whose key is already in `completed` (adding it there once
    written), so re-running this script never duplicates the seeded rows.
    """
    with open(MAIN_CSV, newline="") as f:
        opus_rows = [
            row
            for row in csv.DictReader(f)
            if row["model"] == ALREADY_RUN_MODEL
            and row["blur_type"] == BLUR_TYPE
            and int(row["image_id"]) in image_ids
        ]

    seeded = 0
    with lock:
        for row in opus_rows:
            key = (row["model"], row["blur_type"], int(row["severity"]), int(row["image_id"]))
            if key in completed:
                continue
            pilot_row = {
                "model": row["model"],
                "model_id": MODEL_IDS[ALREADY_RUN_MODEL],
                "blur_type": row["blur_type"],
                "severity": row["severity"],
                "image_id": row["image_id"],
                "true_label": row["true_label"],
                "raw_response": "N/A",
                "prediction": row["prediction"],
                "valid_response": int(row["prediction"] in CIFAR10_LABELS),
                "correct": row["correct"],
                "latency_seconds": "N/A",
                "cost_usd": "N/A",
            }
            append_result(pilot_row, PILOT_CSV, fieldnames=PILOT_FIELDS)
            completed.add(key)
            seeded += 1
    return seeded


def classify_with_metrics(client, model_id, image_array):
    """Like classify_image, but also captures latency, cost, and the raw
    (non-normalized) response text — needed for this pilot's extra columns.
    """
    base64_image = image_array_to_base64(image_array)
    extra_body = dict(get_extra_body(model_id))
    extra_body["usage"] = {"include": True}

    t0 = time.monotonic()
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
    latency = time.monotonic() - t0

    raw_response = response.choices[0].message.content
    prediction = raw_response.strip().lower()
    cost = getattr(response.usage, "cost", None)
    return raw_response, prediction, latency, cost


def main():
    client = get_client()
    labels = load_labels("data/labels.npy")
    corruption = load_corruption(CORRUPTION_FILE)

    image_ids = pick_pilot_image_ids()
    if len(image_ids) < PILOT_IMAGE_COUNT:
        raise RuntimeError(
            f"Expected {PILOT_IMAGE_COUNT} {ALREADY_RUN_MODEL}/{BLUR_TYPE} images in "
            f"{MAIN_CSV}, found {len(image_ids)}."
        )
    print(f"Pilot image IDs (from existing {ALREADY_RUN_MODEL} data): {image_ids}")

    lock = Lock()
    completed = load_completed_keys(PILOT_CSV)
    seeded = seed_opus_rows(image_ids, lock, completed)
    print(f"Seeded {seeded} new {ALREADY_RUN_MODEL} rows into {PILOT_CSV} "
          f"({len(completed)} total already present).")

    tasks = [
        (model_key, model_id, severity, image_id)
        for model_key, model_id in MODEL_IDS.items()
        for severity in SEVERITIES
        for image_id in image_ids
        if (model_key, BLUR_TYPE, severity, image_id) not in completed
    ]
    total_planned = len(MODEL_IDS) * len(SEVERITIES) * len(image_ids)
    print(f"{len(tasks)}/{total_planned} calls remaining (rest already in {PILOT_CSV}).")

    def do_one(task):
        model_key, model_id, severity, image_id = task
        image_array = get_image(corruption, image_id, severity)
        true_label_int = int(labels[image_id])
        true_label = CIFAR10_LABELS[true_label_int]

        try:
            raw_response, prediction, latency, cost = classify_with_metrics(
                client, model_id, image_array
            )
        except Exception as e:
            print(f"  ERROR {model_key} sev{severity} img{image_id}: {e}")
            return

        row = {
            "model": model_key,
            "model_id": model_id,
            "blur_type": BLUR_TYPE,
            "severity": severity,
            "image_id": image_id,
            "true_label": true_label,
            "raw_response": raw_response,
            "prediction": prediction,
            "valid_response": int(prediction in CIFAR10_LABELS),
            "correct": int(prediction == true_label),
            "latency_seconds": round(latency, 3),
            "cost_usd": cost if cost is not None else "N/A",
        }
        with lock:
            append_result(row, PILOT_CSV, fieldnames=PILOT_FIELDS)
            completed.add((model_key, BLUR_TYPE, severity, image_id))

    finished = 0
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        for _ in executor.map(do_one, tasks):
            finished += 1
            if finished % 25 == 0:
                print(f"  {finished}/{len(tasks)}")

    print(f"\nPilot done. {PILOT_CSV} has {len(completed)}/{total_planned} rows.")

    # Real per-call cost, projected to what each model's full 1,400-image
    # run (7,000 calls) would actually cost — not a guess, from this pilot's
    # measured numbers.
    print("\nProjected full-run (7,000 calls) cost per model, from this pilot's real numbers:")
    with open(PILOT_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    for model_key in MODEL_IDS:
        costs = [
            float(r["cost_usd"])
            for r in rows
            if r["model"] == model_key and r["cost_usd"] not in ("N/A", "")
        ]
        if costs:
            avg = sum(costs) / len(costs)
            print(f"  {model_key:15s} avg ${avg:.5f}/call -> ~${avg * 7000:.2f} for 7,000 calls")
        else:
            print(f"  {model_key:15s} no cost data (Opus reused from the main run)")


if __name__ == "__main__":
    main()
