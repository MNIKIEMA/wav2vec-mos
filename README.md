# wav2vec-mos

Wav2Vec2-BERT 2.0 CTC fine-tuning and inference for Moore (Mooré) ASR.
Fine-tunes `facebook/w2v-bert-2.0` on a Hugging Face audio dataset and
pushes the result to the Hub as `burkimbia/wav2vec-mos`.

## Setup

```bash
uv sync
```

Set `HF_TOKEN` (for `push_to_hub`) and `WANDB_API_KEY` (for `report_to:
wandb`) as environment variables or in a `.env` file — both are loaded
automatically via `python-dotenv`.

## Training

```bash
./scripts/train.sh
```

Override any field on the command line, e.g.:

```bash
./scripts/train.sh --num_train_epochs 10
```

## Inference

```bash
./scripts/infer.sh
# or a single local file:
./scripts/infer.sh --audio path/to/clip.wav
```

## Docker

Build:

```bash
docker build -t wav2vec-mos:latest .
```

The image is based on `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel` and
reuses that base image's torch/CUDA install (`uv sync` skips torch,
torchvision, torchaudio, and the `nvidia-*` wheels) to keep the image small.

Run locally:

```bash
docker run --gpus all --rm \
    -e HF_TOKEN -e WANDB_API_KEY \
    wav2vec-mos:latest \
    scripts/train.sh
```

## Vertex AI Training

See [VERTEX.md](VERTEX.md) for mirroring the image to Artifact Registry,
picking a region/accelerator, a `worker-pool-spec.yaml` example, and a GPU
smoke test.
