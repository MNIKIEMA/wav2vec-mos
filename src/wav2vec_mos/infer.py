import json
import sys
from dataclasses import dataclass

import numpy as np
import torch

_MIN_W2V_BERT_SAMPLES = 560


@dataclass
class Wav2VecInferConfig:
    model_name_or_path: str
    model: str = ""
    audio: str = ""
    dataset: str = ""
    dataset_config: str = ""
    split: str = "test"
    audio_column: str = "audio"
    output_file: str = ""
    batch_size: int = 16
    device: str = "cuda"
    sampling_rate: int = 16000
    min_audio_seconds: float = 0.1


def _extract_audio_array(audio: dict) -> np.ndarray:
    array = np.asarray(audio["array"], dtype=np.float32)
    if array.ndim > 1:
        array = array.mean(axis=1)
    return np.asarray(array, dtype=np.float32)


def _decode_batch(model, processor, audio_arrays: list, device: str) -> list[str]:
    inputs = processor(
        audio_arrays,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    ).to(device)
    with torch.inference_mode():
        logits = model(**inputs).logits
    pred_ids = logits.argmax(dim=-1)
    return processor.batch_decode(pred_ids)


def infer(cfg: Wav2VecInferConfig) -> None:
    from transformers import AutoProcessor, Wav2Vec2BertForCTC

    device = cfg.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    processor = AutoProcessor.from_pretrained(cfg.model_name_or_path)
    model = Wav2Vec2BertForCTC.from_pretrained(cfg.model_name_or_path, torch_dtype=dtype).to(device).eval()

    out_stream = open(cfg.output_file, "w", encoding="utf-8") if cfg.output_file else None

    def _emit(record: dict) -> None:
        print(json.dumps(record, ensure_ascii=False), file=out_stream or sys.stdout)

    from wav2vec_mos.profiling import profile_inference

    try:
        if cfg.audio:
            import torchaudio

            wav, sr = torchaudio.load(cfg.audio)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            audio_array = wav.squeeze(0).numpy()
            audio_duration_s = len(audio_array) / 16000
            with profile_inference(audio_duration_s) as prof:
                results = _decode_batch(model, processor, [audio_array], device)
            _emit(
                {
                    "transcription": results[0],
                    "latency_s": prof.latency_s,
                    "rtf": prof.rtf,
                    "peak_gpu_mb": prof.peak_gpu_mb,
                }
            )
            return

        from datasets import Audio, load_dataset

        ds = load_dataset(cfg.dataset, cfg.dataset_config or None, split=cfg.split)
        ds = ds.cast_column(cfg.audio_column, Audio(sampling_rate=cfg.sampling_rate, decode=True))
        has_id = "id" in ds.column_names
        min_samples = max(_MIN_W2V_BERT_SAMPLES, int(cfg.min_audio_seconds * cfg.sampling_rate))

        for i in range(0, len(ds), cfg.batch_size):
            batch = ds[i : i + cfg.batch_size]
            audio_arrays = []
            records = []
            for j, audio in enumerate(batch[cfg.audio_column]):
                record = {"index": i + j}
                if has_id:
                    record["id"] = batch["id"][j]
                try:
                    audio_array = _extract_audio_array(audio)
                    if len(audio_array) < min_samples:
                        raise ValueError(f"audio row is too short: {len(audio_array)} samples")
                except Exception as exc:
                    record["error"] = f"skipped unreadable audio: {exc}"
                    _emit(record)
                    continue
                audio_arrays.append(audio_array)
                records.append(record)

            if not audio_arrays:
                continue

            audio_duration_s = sum(len(a) for a in audio_arrays) / cfg.sampling_rate
            with profile_inference(audio_duration_s) as prof:
                texts = _decode_batch(model, processor, audio_arrays, device)
            for record, text in zip(records, texts, strict=True):
                record.update(
                    {
                        "transcription": text,
                        "latency_s": prof.latency_s,
                        "rtf": prof.rtf,
                        "peak_gpu_mb": prof.peak_gpu_mb,
                    }
                )
                _emit(record)
    finally:
        if out_stream:
            out_stream.close()
