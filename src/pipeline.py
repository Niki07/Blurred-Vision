"""Blur-robustness data-collection pipeline for the paper.

This is the methodology pipeline: run_blur_experiment(blur_name) classifies
a fixed sample of CIFAR-10-C images across every configured model and
severity level for one blur corruption (motion/glass/defocus), appending one
row per (model, severity, image) test to its own results CSV as each result
completes — never buffered to the end of the run.

Each blur type gets its own folder and CSV (results/<blur>/results_<blur>.csv)
so runs never overwrite or mix with each other:

    run_blur_experiment("motion")    -> results/motion/results_motion.csv
    run_blur_experiment("defocus")   -> results/defocus/results_defocus.csv
    run_blur_experiment("glass")     -> results/glass/results_glass.csv

Resume-safe: on every start (including a restart after a crash, a stopped
run, or running out of API credit), it re-reads that blur's results CSV
first and skips any (model, blur_type, severity, image_id) already recorded
there. A test is only ever paid for once.

See ENGINEERING_DECISIONS.md for why this schema/design looks the way it
does, and Research context.md for the overall study design.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest

from src.classify import classify_image_with_metrics
from src.client import get_client, get_remaining_credits
from src.config import CIFAR10_LABELS, DETAILED_RESULT_FIELDS, MODEL_IDS, SEVERITIES
from src.data import build_fixed_sample, get_image, load_corruption, load_labels
from src.persist import append_result, load_completed_keys

# ============================== DEFAULT CONFIGURATION ==============================
# Defaults for run_blur_experiment()'s keyword arguments. Override per-call
# rather than editing these, e.g. run_blur_experiment("glass", concurrency=20).

# Which models to test by default — a {short_name: openrouter_model_id} dict.
DEFAULT_MODELS = MODEL_IDS

# Which severities to test by default. All 5 (the full research design).
DEFAULT_SEVERITIES = SEVERITIES

# Default fixed sample: first N base images per class, in dataset order (not
# random) — see ENGINEERING_DECISIONS.md. Pass image_ids= to override with an
# explicit list of base image indices instead, e.g. for a small manual test.
DEFAULT_IMAGES_PER_CLASS = 140

# How many calls to run at once, spread across all models (so this isn't
# hammering any single provider's rate limit — see ENGINEERING_DECISIONS.md).
# 20 as a starting default: fast enough, and a bit more conservative than the
# 40 used for motion's first pass, which needed lowering to 10 partway
# through as balance dropped and OpenRouter's in-flight budget cap tightened.
DEFAULT_CONCURRENCY = 20

# Live balance safeguard defaults: a background thread checks your real
# OpenRouter balance every BALANCE_CHECK_INTERVAL_SECONDS and stops any calls
# not already in flight once it drops below MIN_BALANCE_USD.
DEFAULT_MIN_BALANCE_USD = 1.0
DEFAULT_BALANCE_CHECK_INTERVAL_SECONDS = 20
# =====================================================================================


def build_sample(labels, images_per_class, image_ids):
    if image_ids is not None:
        return list(image_ids)
    return build_fixed_sample(labels, images_per_class)


def watch_balance(stop_event, min_balance_usd, check_interval_seconds):
    """Background thread: stop new calls once real balance drops too low.

    Doesn't cancel calls already in flight — just stops new ones from being
    made, so a low-balance stop is clean rather than an abrupt kill.
    """
    while not stop_event.is_set():
        try:
            remaining = get_remaining_credits()
            if remaining < min_balance_usd:
                print(
                    f"\nLOW BALANCE: ${remaining:.2f} remaining "
                    f"(below ${min_balance_usd:.2f} threshold). Stopping new calls — "
                    f"add funds and rerun to pick up where this left off."
                )
                stop_event.set()
                return
        except Exception as e:
            print(f"  (balance check failed, will retry: {e})")
        stop_event.wait(check_interval_seconds)


def run_blur_experiment(
    blur_name,
    models_to_run=None,
    severities_to_run=None,
    images_per_class=DEFAULT_IMAGES_PER_CLASS,
    image_ids=None,
    concurrency=DEFAULT_CONCURRENCY,
    min_balance_usd=DEFAULT_MIN_BALANCE_USD,
    balance_check_interval_seconds=DEFAULT_BALANCE_CHECK_INTERVAL_SECONDS,
):
    """Run (or resume) the full data-collection pass for one blur corruption.

    blur_name: "motion", "glass", or "defocus" — derives the CIFAR-10-C file
    (data/<blur_name>_blur.npy), the blur_type recorded in each row
    (<blur_name>_blur), and the output CSV (results/<blur_name>/results_<blur_name>.csv).
    """
    models_to_run = models_to_run if models_to_run is not None else DEFAULT_MODELS
    severities_to_run = severities_to_run if severities_to_run is not None else DEFAULT_SEVERITIES

    blur_type = f"{blur_name}_blur"
    corruption_file = f"data/{blur_type}.npy"
    results_csv = f"results/{blur_name}/results_{blur_name}.csv"

    client = get_client()
    labels = load_labels("data/labels.npy")
    corruption = load_corruption(corruption_file)
    sample = build_sample(labels, images_per_class, image_ids)
    completed = load_completed_keys(results_csv)
    lock = threading.Lock()

    starting_balance = get_remaining_credits()
    print(f"[{blur_name}] Starting balance: ${starting_balance:.2f} (stop threshold: ${min_balance_usd:.2f})")
    if starting_balance < min_balance_usd:
        print(f"[{blur_name}] Already below the stop threshold — add funds before running.")
        return

    stop_event = threading.Event()
    watcher = threading.Thread(
        target=watch_balance,
        args=(stop_event, min_balance_usd, balance_check_interval_seconds),
        daemon=True,
    )
    watcher.start()

    # Interleaved round-robin across models (not grouped model-by-model) so
    # `concurrency` workers always pull a mix of cheap and expensive models
    # at once, rather than bursting dozens of concurrent calls to one
    # flagship model — that burst tripped OpenRouter's per-request in-flight
    # budget cap (a 402, distinct from actually running out of balance)
    # during the motion run. See ENGINEERING_DECISIONS.md.
    per_model_tasks = [
        [
            (model_key, model_id, severity, image_id)
            for severity in severities_to_run
            for image_id in sample
        ]
        for model_key, model_id in models_to_run.items()
    ]
    tasks = [
        task
        for round_ in zip_longest(*per_model_tasks)
        for task in round_
        if task is not None and (task[0], blur_type, task[2], task[3]) not in completed
    ]
    total_planned = len(models_to_run) * len(severities_to_run) * len(sample)
    print(
        f"[{blur_name}] Blur type: {blur_type} | models: {len(models_to_run)} | "
        f"severities: {len(severities_to_run)} | sample: {len(sample)} images | "
        f"concurrency: {concurrency}"
    )
    print(f"[{blur_name}] {len(tasks)}/{total_planned} calls remaining (rest already in {results_csv}).")

    def do_one(task):
        if stop_event.is_set():
            return
        model_key, model_id, severity, image_id = task
        image_array = get_image(corruption, image_id, severity)
        true_label = CIFAR10_LABELS[int(labels[image_id])]

        raw_response = prediction = latency_seconds = cost_usd = None
        for attempt in range(3):
            try:
                raw_response, prediction, latency_seconds, cost_usd = classify_image_with_metrics(
                    client, model_id, image_array
                )
                break
            except Exception as e:
                if attempt == 2:
                    print(f"[{blur_name}]   ERROR {model_key} sev{severity} img{image_id}: {e}")
                    return
                time.sleep(5 * (attempt + 1))

        row = {
            "model": model_key,
            "model_id": model_id,
            "blur_type": blur_type,
            "severity": severity,
            "image_id": image_id,
            "true_label": true_label,
            "raw_response": raw_response,
            "prediction": prediction,
            "valid_response": int(prediction in CIFAR10_LABELS),
            "correct": int(prediction == true_label),
            "latency_seconds": round(latency_seconds, 3),
            "cost_usd": cost_usd if cost_usd is not None else "N/A",
        }
        with lock:
            append_result(row, results_csv, fieldnames=DETAILED_RESULT_FIELDS)
            completed.add((model_key, blur_type, severity, image_id))

    finished = 0
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for _ in executor.map(do_one, tasks):
            finished += 1
            if finished % 500 == 0:
                print(f"[{blur_name}]   {finished}/{len(tasks)}")

    stop_event.set()  # let the watcher thread exit
    status = "stopped early on low balance" if len(completed) < total_planned else "all done"
    print(f"\n[{blur_name}] {status}. {results_csv} has {len(completed)}/{total_planned} rows.")


if __name__ == "__main__":
    import sys

    run_blur_experiment(sys.argv[1] if len(sys.argv) > 1 else "motion")
