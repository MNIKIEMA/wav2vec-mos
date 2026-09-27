import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass
class InferenceProfile:
    latency_s: float = 0.0
    rtf: float | None = None
    peak_gpu_mb: float | None = None


@contextmanager
def profile_inference(
    audio_duration_s: float | None = None, track_gpu: bool = True
) -> Iterator[InferenceProfile]:
    """Measure wall-clock latency, RTF, and peak GPU memory around an inference call.

    Peak memory comes from the torch CUDA allocator, so pass ``track_gpu=False`` for runtimes
    (e.g. onnxruntime) whose allocations torch cannot see.
    """
    prof = InferenceProfile()
    has_cuda = track_gpu and torch.cuda.is_available()
    if has_cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        yield prof
    finally:
        prof.latency_s = time.perf_counter() - t0
        if audio_duration_s:
            prof.rtf = prof.latency_s / audio_duration_s
        if has_cuda:
            prof.peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024**2)
