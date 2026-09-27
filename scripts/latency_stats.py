"""Latency stats for `wav2vec-mos infer` JSONL outputs, and transcription agreement between runs.

Usage: uv run python scripts/latency_stats.py predictions/run_a.jsonl [predictions/run_b.jsonl ...]
"""

import json
import sys
from pathlib import Path

import numpy as np

runs = {}
for path in sys.argv[1:]:
    name = Path(path).stem
    records = [json.loads(line) for line in open(path)]
    runs[name] = records

    ok = [r for r in records if "error" not in r]
    # Every record in a batch carries the same latency_s/rtf, so collapse them back to batches.
    batches = list(dict.fromkeys((r["latency_s"], r["rtf"]) for r in ok))
    lat = np.array([b[0] for b in batches])
    audio = np.array([b[0] / b[1] for b in batches])
    rtf = lat / audio

    def row(label, lat, audio, rtf):
        p50, p90, p95 = np.percentile(lat, [50, 90, 95])
        print(
            f"  {label:<14} batches={len(lat):>2}  audio={audio.sum():6.1f}s  total={lat.sum():6.1f}s  "
            f"RTF={lat.sum() / audio.sum():.3f}  batch latency: mean={lat.mean():.2f} p50={p50:.2f} "
            f"p90={p90:.2f} p95={p95:.2f} max={lat.max():.2f}s  RTF p50={np.median(rtf):.3f} p90={np.percentile(rtf, 90):.3f}"
        )

    print(f"{name}: {len(records)} records, {len(records) - len(ok)} skipped")
    row("all batches", lat, audio, rtf)
    row("excl. first", lat[1:], audio[1:], rtf[1:])

names = list(runs)
ref = {r["index"]: r.get("transcription") for r in runs[names[0]]}
for other in names[1:]:
    same = sum(ref[r["index"]] == r.get("transcription") for r in runs[other])
    print(f"transcriptions identical {names[0]} vs {other}: {same}/{len(runs[other])}")
