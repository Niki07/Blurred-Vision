"""Motion-blur data-collection pipeline for the blur-robustness paper.

This is the methodology pipeline: it classifies a fixed sample of CIFAR-10-C
images across every configured model and severity level, and appends one row
per (model, severity, image) test to RESULTS_CSV as each result completes —
never buffered to the end of the run.

Resume-safe: on every start (including a restart after a crash, a stopped
run, or running out of API credit), it re-reads RESULTS_CSV first and skips
any (model, blur_type, severity, image_id) already recorded there. A test is
only ever paid for once.

Starts from scratch: this script never reads or reuses results.csv or
results_mini.csv from earlier runs — only RESULTS_CSV below.

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

# ============================== RUN CONFIGURATION ==============================
# Edit these to control exactly what this run covers. Nothing below this
# block needs to change for a normal configuration change.

# Which models to test — a {short_name: openrouter_model_id} dict. Defaults
# to the full roster (src/config.py:MODEL_IDS). Trim it to run a subset,
# e.g. MODELS_TO_RUN = {"claude_opus": MODEL_IDS["claude_opus"]}.
MODELS_TO_RUN = MODEL_IDS

# Which corruption file this run covers. Only motion_blur for now (see
# ENGINEERING_DECISIONS.md) — glass_blur/defocus_blur are separate future
# runs against their own RESULTS_CSV, not mixed into this one.
BLUR_TYPE = "motion_blur"
CORRUPTION_FILE = "data/motion_blur.npy"

# Which severities to test. Defaults to all 5 (the full research design).
SEVERITIES_TO_RUN = SEVERITIES

# Which base images to test. Leave IMAGE_IDS as None to auto-build a fixed
# sample of IMAGES_PER_CLASS images per CIFAR-10 class (the first N found in
# dataset order, not random — see ENGINEERING_DECISIONS.md). Set IMAGE_IDS to
# an explicit list of base image indices instead to override this entirely,
# e.g. IMAGE_IDS = [0, 1, 2, 3, 4] for a small manual test.
IMAGES_PER_CLASS = 140
IMAGE_IDS = None

# Output file. Brand new — never reads or overwrites results.csv or
# results_mini.csv from earlier runs.
RESULTS_CSV = "results/results_motion.csv"

# How many calls to run at once, spread across MODELS_TO_RUN (so this isn't
# hammering any single provider's rate limit — see ENGINEERING_DECISIONS.md).
# Lowered from 40 to 10 (2026-08-31): OpenRouter's per-account in-flight
# budget cap shrinks as real balance drops, so the same concurrency that was
# safe at $21 started tripping 402 in_flight_budget_exhausted errors more
# often as balance fell past ~$8. Interleaving (below) fixed the "40
# concurrent calls to one expensive model" case; this fixes the "any 40
# concurrent calls at all once balance is low" case.
CONCURRENCY = 10

# Live balance safeguard: a background thread checks your real OpenRouter
# balance (not an estimate) every BALANCE_CHECK_INTERVAL_SECONDS, and stops
# any calls not already in flight once it drops below MIN_BALANCE_USD. See
# ENGINEERING_DECISIONS.md.
MIN_BALANCE_USD = 1.0
BALANCE_CHECK_INTERVAL_SECONDS = 20
# ================================================================================


def build_sample(labels):
    if IMAGE_IDS is not None:
        return list(IMAGE_IDS)
    return build_fixed_sample(labels, IMAGES_PER_CLASS)


def watch_balance(stop_event):
    """Background thread: stop new calls once real balance drops too low.

    Doesn't cancel calls already in flight — just stops new ones from being
    made, so a low-balance stop is clean rather than an abrupt kill.
    """
    while not stop_event.is_set():
        try:
            remaining = get_remaining_credits()
            if remaining < MIN_BALANCE_USD:
                print(
                    f"\nLOW BALANCE: ${remaining:.2f} remaining "
                    f"(below ${MIN_BALANCE_USD:.2f} threshold). Stopping new calls — "
                    f"add funds and rerun to pick up where this left off."
                )
                stop_event.set()
                return
        except Exception as e:
            print(f"  (balance check failed, will retry: {e})")
        stop_event.wait(BALANCE_CHECK_INTERVAL_SECONDS)


def main():
    client = get_client()
    labels = load_labels("data/labels.npy")
    corruption = load_corruption(CORRUPTION_FILE)
    sample = build_sample(labels)
    completed = load_completed_keys(RESULTS_CSV)
    lock = threading.Lock()

    starting_balance = get_remaining_credits()
    print(f"Starting balance: ${starting_balance:.2f} (stop threshold: ${MIN_BALANCE_USD:.2f})")
    if starting_balance < MIN_BALANCE_USD:
        print("Already below the stop threshold — add funds before running.")
        return

    stop_event = threading.Event()
    watcher = threading.Thread(target=watch_balance, args=(stop_event,), daemon=True)
    watcher.start()

    # Interleaved round-robin across models (not grouped model-by-model) so
    # CONCURRENCY workers always pull a mix of cheap and expensive models at
    # once, rather than bursting dozens of concurrent calls to one flagship
    # model — that burst tripped OpenRouter's per-request in-flight budget
    # cap (a 402, distinct from actually running out of balance) during the
    # first live run. See ENGINEERING_DECISIONS.md.
    per_model_tasks = [
        [
            (model_key, model_id, severity, image_id)
            for severity in SEVERITIES_TO_RUN
            for image_id in sample
        ]
        for model_key, model_id in MODELS_TO_RUN.items()
    ]
    tasks = [
        task
        for round_ in zip_longest(*per_model_tasks)
        for task in round_
        if task is not None and (task[0], BLUR_TYPE, task[2], task[3]) not in completed
    ]
    total_planned = len(MODELS_TO_RUN) * len(SEVERITIES_TO_RUN) * len(sample)
    print(
        f"Blur type: {BLUR_TYPE} | models: {len(MODELS_TO_RUN)} | "
        f"severities: {len(SEVERITIES_TO_RUN)} | sample: {len(sample)} images | "
        f"concurrency: {CONCURRENCY}"
    )
    print(f"{len(tasks)}/{total_planned} calls remaining (rest already in {RESULTS_CSV}).")

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
                    print(f"  ERROR {model_key} sev{severity} img{image_id}: {e}")
                    return
                time.sleep(5 * (attempt + 1))

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
            "latency_seconds": round(latency_seconds, 3),
            "cost_usd": cost_usd if cost_usd is not None else "N/A",
        }
        with lock:
            append_result(row, RESULTS_CSV, fieldnames=DETAILED_RESULT_FIELDS)
            completed.add((model_key, BLUR_TYPE, severity, image_id))

    finished = 0
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        for _ in executor.map(do_one, tasks):
            finished += 1
            if finished % 500 == 0:
                print(f"  {finished}/{len(tasks)}")

    stop_event.set()  # let the watcher thread exit
    status = "stopped early on low balance" if len(completed) < total_planned else "all done"
    print(f"\n{status}. {RESULTS_CSV} has {len(completed)}/{total_planned} rows.")


if __name__ == "__main__":
    main()
