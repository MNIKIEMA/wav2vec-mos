import json
import sys
from dataclasses import dataclass
from pathlib import Path

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
    onnx: bool = False
    onnx_file: str = "model.onnx"


def _extract_audio_array(audio: dict) -> np.ndarray:
    array = np.asarray(audio["array"], dtype=np.float32)
    if array.ndim > 1:
        array = array.mean(axis=1)
    return np.asarray(array, dtype=np.float32)


def _torch_predictor(cfg: Wav2VecInferConfig, processor, device: str):
    from transformers import Wav2Vec2BertForCTC

    model = Wav2Vec2BertForCTC.from_pretrained(cfg.model_name_or_path, torch_dtype=torch.float32)
    model = model.to(device).eval()

    def predict(audio_arrays: list):
        inputs = processor(audio_arrays, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            return model(**inputs).logits.argmax(dim=-1)

    return predict


def _onnx_predictor(cfg: Wav2VecInferConfig, processor, device: str):
    import onnxruntime as ort

    # onnx_file is relative to model_name_or_path, e.g. "onnx/model_int8.onnx" on the Hub.
    model_path = Path(cfg.model_name_or_path) / cfg.onnx_file
    if not model_path.exists():
        from huggingface_hub import snapshot_download

        # The trailing * also fetches fp32 external data (model.onnx.data).
        repo_dir = snapshot_download(cfg.model_name_or_path, allow_patterns=[f"{cfg.onnx_file}*"])
        model_path = Path(repo_dir) / cfg.onnx_file

    providers = ["CPUExecutionProvider"]
    if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    session = ort.InferenceSession(str(model_path), providers=providers)

    def predict(audio_arrays: list):
        inputs = processor(audio_arrays, sampling_rate=16000, return_tensors="np", padding=True)
        (logits,) = session.run(
            None,
            {
                "input_features": inputs["input_features"].astype(np.float32),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            },
        )
        return logits.argmax(axis=-1)

    return predict


def _decode_batch(predict, processor, audio_arrays: list) -> list[str]:
    return processor.batch_decode(predict(audio_arrays))


def infer(cfg: Wav2VecInferConfig) -> None:
    from transformers import AutoProcessor

    device = cfg.device if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(cfg.model_name_or_path)
    make_predictor = _onnx_predictor if cfg.onnx else _torch_predictor
    predict = make_predictor(cfg, processor, device)

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
            with profile_inference(audio_duration_s, track_gpu=not cfg.onnx) as prof:
                results = _decode_batch(predict, processor, [audio_array])
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
            with profile_inference(audio_duration_s, track_gpu=not cfg.onnx) as prof:
                texts = _decode_batch(predict, processor, audio_arrays)
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
