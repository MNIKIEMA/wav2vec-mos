#!/usr/bin/env bash
set -euo pipefail

uv run wav2vec-mos train \
    --model_name_or_path facebook/w2v-bert-2.0 \
    --dataset burkimbia/speech-dataset-processed \
    --dataset_config "" \
    --length_column_name duration \
    --output_dir outputs/wav2vec-mos \
    --hub_model_id burkimbia/wav2vec-mos \
    --hub_private \
    --push_to_hub false \
    --wandb_project "Wav2Vec-BERT-2.0" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --learning_rate 2e-5 \
    --freeze_feature_encoder \
    --gradient_checkpointing \
    --preprocessing_num_proc 4 \
    "$@"
