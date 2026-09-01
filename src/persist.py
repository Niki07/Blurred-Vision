"""Append-as-you-go result persistence.

Critical requirement (see Research context.md): results must survive a crash,
disconnect, or credit cutoff mid-run. Every row is flushed to disk immediately
after it's produced, never buffered until the end of a run.
"""

import csv
import os

from src.config import RESULT_FIELDS


def append_result(row, csv_path, fieldnames=RESULT_FIELDS):
    """Append one result row to csv_path, creating it with a header if needed."""
    file_exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def load_completed_keys(csv_path):
    """Return the (model, blur_type, severity, image_id) keys already in csv_path.

    Used to make a run resumable: skip any call whose result is already on disk,
    whether the previous run was interrupted or manually paused between models.
    Works for any CSV that has these four columns, regardless of what other
    columns it also carries.
    """
    completed = set()
    if not os.path.exists(csv_path):
        return completed

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            completed.add(
                (row["model"], row["blur_type"], int(row["severity"]), int(row["image_id"]))
            )
    return completed
