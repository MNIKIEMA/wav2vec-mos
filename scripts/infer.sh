#!/usr/bin/env bash
set -euo pipefail

uv run --no-sync wav2vec-mos infer \
    --model_name_or_path burkimbia/wav2vec-mos \
    --dataset burkimbia/asr-benchmark-public \
    --dataset_config default \
    --split train \
    --audio_column audio \
    --output_file predictions_asr_public.jsonl \
    --batch_size 16 \
    "$@"
