"""Benchmark report consolidation."""

from __future__ import annotations

from pathlib import Path
import csv
import json
from statistics import mean


def consolidate_benchmark(input_csv: str | Path, output_json: str | Path) -> dict:
    rows = []
    with Path(input_csv).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    if not rows:
        raise ValueError("benchmark CSV has no rows")

    def values(name: str):
        return [float(row[name]) for row in rows if row.get(name) not in (None, "")]

    report = {
        "row_count": len(rows),
        "bd_rate_mean": mean(values("bd_rate")) if values("bd_rate") else None,
        "delta_psnr_mean": mean(values("delta_psnr")) if values("delta_psnr") else None,
        "time_reduction_mean": mean(values("time_reduction")) if values("time_reduction") else None,
        "rows": rows,
    }
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
