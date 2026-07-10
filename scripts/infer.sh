#!/usr/bin/env bash
set -euo pipefail

uv run wav2vec-mos infer \
    --model_name_or_path outputs/wav2vec-mos \
    --dataset burkimbia/asr-benchmark-public \
    --dataset_config default \
    --split train \
    --audio_column audio \
    --output_file predictions_asr_public.jsonl \
    --batch_size 16 \
    "$@"
