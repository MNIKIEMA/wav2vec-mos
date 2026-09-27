import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Wav2VecExportConfig:
    model_name_or_path: str
    output_dir: str = "onnx"
    opset_version: int = 18
    quantize: bool = False


def export_onnx(cfg: Wav2VecExportConfig) -> None:
    from transformers import AutoProcessor, Wav2Vec2BertForCTC

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(cfg.model_name_or_path)
    model = Wav2Vec2BertForCTC.from_pretrained(cfg.model_name_or_path, torch_dtype=torch.float32).eval()

    # The feature extractor and CTC decoding stay outside the graph; only the network is exported.
    feature_size = processor.feature_extractor.feature_size * processor.feature_extractor.stride
    # Trace with a padded batch of 2: a batch of 1 lets the exporter bake size-1 broadcasts into the
    # graph, which then fails at runtime on real batches.
    input_features = torch.randn(2, 200, feature_size)
    attention_mask = torch.ones(2, 200, dtype=torch.long)
    attention_mask[1, 150:] = 0

    def _export(path: Path) -> None:
        torch.onnx.export(
            model,
            (input_features, attention_mask),
            path,
            input_names=["input_features", "attention_mask"],
            output_names=["logits"],
            opset_version=cfg.opset_version,
            dynamic_shapes=({0: "batch", 1: "frames"}, {0: "batch", 1: "frames"}),
            dynamo=True,
        )

    if cfg.quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        # int8 weights, activations quantized on the fly. Only MatMul is quantized: it holds almost all
        # the weights, and ConvInteger has poor kernel coverage in onnxruntime.
        with tempfile.TemporaryDirectory() as tmp:
            fp32_path = Path(tmp) / "model.onnx"
            _export(fp32_path)
            quantize_dynamic(
                fp32_path,
                output_dir / "model.onnx",
                op_types_to_quantize=["MatMul"],
                weight_type=QuantType.QInt8,
            )
    else:
        _export(output_dir / "model.onnx")

    processor.save_pretrained(output_dir)
    kind = "int8-quantized ONNX" if cfg.quantize else "ONNX"
    print(f"Exported {kind} model and processor to {output_dir}")
