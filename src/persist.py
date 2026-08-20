"""Append-as-you-go result persistence.

Critical requirement (see Research context.md): results must survive a crash,
disconnect, or credit cutoff mid-run. Every row is flushed to disk immediately
after it's produced, never buffered until the end of a run.
"""

import csv
import os

from src.config import RESULT_FIELDS


def append_result(row, csv_path):
    """Append one result row to csv_path, creating it with a header if needed."""
    file_exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())
