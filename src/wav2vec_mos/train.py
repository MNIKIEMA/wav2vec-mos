import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv

# Built from numeric codepoints, not typed glyphs, so visually-identical
# characters (e.g. curly vs straight quotes) can't silently duplicate/collide.
_PUNCT_CODEPOINTS_TO_REMOVE = [
    0x22,
    0x27,  # straight double/single quote
    0x2018,
    0x2019,
    0x201C,
    0x201D,  # curly single/double quotes
    0x2010,
    0x2011,
    0x2012,
    0x2013,
    0x2014,
    0x2015,  # hyphen/dash variants
    0x00AB,
    0x00BB,
    0x2039,
    0x203A,  # guillemets, double and single
    0x2026,  # ellipsis
]
_CHARS_TO_REMOVE_RE = re.compile(
    "[" + r"\d\,\?\.\!\;\:\%\(\)\*\+" + "".join(chr(c) for c in _PUNCT_CODEPOINTS_TO_REMOVE) + "]"
)

load_dotenv()


@dataclass
class Wav2VecConfig:
    model_name_or_path: str
    dataset: str
    output_dir: str
    model: str = ""
    dataset_config: str = ""
    train_split: str = "train"
    eval_split: str = "test"
    text_column: str = "text"
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    save_steps: int = 600
    eval_steps: int = 300
    logging_steps: int = 100
    save_total_limit: int = 2
    freeze_feature_encoder: bool = True
    gradient_checkpointing: bool = True
    fp16: bool = True
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    feat_proj_dropout: float = 0.0
    mask_time_prob: float = 0.0
    layerdrop: float = 0.0
    push_to_hub: bool = True
    hub_model_id: str = ""
    hub_private: bool = False
    report_to: str = "wandb"
    run_name: str = ""
    wandb_project: str = ""
    sampling_rate: int = 16000
    length_column_name: str = "duration"
    preprocessing_num_proc: int = 1
    dataloader_num_workers: int = 4


def _clean_text(batch: dict, col: str) -> dict:
    text = unicodedata.normalize("NFKC", batch[col])
    batch[col] = _CHARS_TO_REMOVE_RE.sub("", text).lower()
    return batch


def _extract_chars(batch: dict, col: str) -> dict:
    all_text = " ".join(batch[col])
    return {"vocab": [list(set(all_text))], "all_text": [all_text]}


def _build_vocab_dict(train_vocab, test_vocab) -> dict:
    vocab_list = sorted(set(train_vocab["vocab"][0]) | set(test_vocab["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    return vocab_dict


@dataclass
class _DataCollatorCTCWithPadding:
    processor: Any
    padding: bool | str = True

    def __call__(self, features: list[dict[str, list[int] | torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def _make_compute_metrics(processor):
    def _compute(pred):
        import jiwer

        pred_ids = pred.predictions.argmax(-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        return {"wer": jiwer.wer(label_str, pred_str)}

    return _compute


def _build_processor(train_ds, test_ds, cfg: Wav2VecConfig, vocab_path: Path):
    from transformers import AutoFeatureExtractor, Wav2Vec2BertProcessor, Wav2Vec2CTCTokenizer

    extract_fn = lambda b: _extract_chars(b, cfg.text_column)  # noqa: E731

    train_vocab = train_ds.map(
        extract_fn, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=train_ds.column_names
    )
    test_vocab = test_ds.map(
        extract_fn, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=test_ds.column_names
    )

    vocab_dict = _build_vocab_dict(train_vocab, test_vocab)
    vocab_path.write_text(json.dumps(vocab_dict, ensure_ascii=False, indent=2))

    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path), unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.model_name_or_path)
    return Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def _load_model(cfg: Wav2VecConfig, processor):
    from transformers import Wav2Vec2BertForCTC

    return Wav2Vec2BertForCTC.from_pretrained(
        cfg.model_name_or_path,
        attention_dropout=cfg.attention_dropout,
        hidden_dropout=cfg.hidden_dropout,
        feat_proj_dropout=cfg.feat_proj_dropout,
        mask_time_prob=cfg.mask_time_prob,
        layerdrop=cfg.layerdrop,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        add_adapter=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )


def _prepare_dataset(batch: dict, processor, cfg: Wav2VecConfig) -> dict:
    audio = batch["audio"]
    # New datasets/TorchCodec format use AudioDecoder object
    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        audio_array = samples.data.squeeze().numpy()
        sampling_rate = samples.sample_rate
    else:
        audio_array = audio["array"]
        sampling_rate = audio.get("sampling_rate", cfg.sampling_rate)

    batch["input_features"] = processor(
        audio_array,
        sampling_rate=sampling_rate,
    ).input_features[0]
    batch["labels"] = processor(text=batch[cfg.text_column]).input_ids
    return batch


def _prepare_split(dataset, processor, cfg: Wav2VecConfig):
    prepare_fn = lambda b: _prepare_dataset(b, processor, cfg)  # noqa: E731
    num_proc = cfg.preprocessing_num_proc if cfg.preprocessing_num_proc and cfg.preprocessing_num_proc > 1 else None
    dataset = dataset.map(prepare_fn, num_proc=num_proc)
    keep_columns = {"input_features", "labels", cfg.length_column_name}
    remove_columns = [col for col in dataset.column_names if col not in keep_columns]
    if remove_columns:
        dataset = dataset.remove_columns(remove_columns)
    return dataset


def train(cfg: Wav2VecConfig) -> None:
    from datasets import Audio, load_dataset
    from transformers import Trainer, TrainingArguments

    from wav2vec_mos.utils import get_run_name

    if cfg.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)

    load_kwargs: dict = {"path": cfg.dataset}
    if cfg.dataset_config:
        load_kwargs["name"] = cfg.dataset_config
    raw = load_dataset(**load_kwargs)
    raw = raw.cast_column("audio", Audio(sampling_rate=cfg.sampling_rate, decode=True))
    raw = raw.map(lambda b: _clean_text(b, cfg.text_column))

    if cfg.eval_split not in raw:
        import warnings

        train_size = len(raw[cfg.train_split])
        test_size = 2000 if train_size >= 10000 else 0.05
        warnings.warn(
            f"Split '{cfg.eval_split}' not found in dataset. "
            f"Sampling {'2000 rows' if train_size >= 10000 else '5%'} from '{cfg.train_split}' for validation.",
            UserWarning,
            stacklevel=2,
        )
        split = raw[cfg.train_split].train_test_split(test_size=test_size, seed=42)
        raw[cfg.train_split] = split["train"]
        raw[cfg.eval_split] = split["test"]

    vocab_path = Path(cfg.output_dir) / "vocab.json"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    processor = _build_processor(raw[cfg.train_split], raw[cfg.eval_split], cfg, vocab_path)

    raw_hub_id = cfg.hub_model_id
    if raw_hub_id and "/" not in raw_hub_id:
        from huggingface_hub import whoami

        raw_hub_id = f"{whoami()['name']}/{raw_hub_id}"
    hub_model_id = raw_hub_id or None

    run_name = cfg.run_name or get_run_name(
        model_id=cfg.model_name_or_path,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.per_device_train_batch_size,
        accumulation_steps=cfg.gradient_accumulation_steps,
        num_epochs=cfg.num_train_epochs,
        tags=["frozen"] if cfg.freeze_feature_encoder else ["full"],
    )

    if cfg.push_to_hub:
        processor.push_to_hub(hub_model_id or cfg.output_dir, private=cfg.hub_private)

    train_ds = _prepare_split(raw[cfg.train_split], processor, cfg)
    test_ds = _prepare_split(raw[cfg.eval_split], processor, cfg)

    model = _load_model(cfg, processor)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        hub_model_id=hub_model_id,
        train_sampling_strategy="group_by_length",
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        eval_strategy="steps",
        num_train_epochs=cfg.num_train_epochs,
        gradient_checkpointing=cfg.gradient_checkpointing,
        fp16=cfg.fp16,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        logging_steps=cfg.logging_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        save_total_limit=cfg.save_total_limit,
        push_to_hub=cfg.push_to_hub,
        hub_private_repo=cfg.hub_private,
        report_to=cfg.report_to,
        run_name=run_name,
        dataloader_num_workers=cfg.dataloader_num_workers,
        length_column_name=cfg.length_column_name,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        data_collator=_DataCollatorCTCWithPadding(processor=processor),
        args=training_args,
        compute_metrics=_make_compute_metrics(processor),
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=processor,
    )

    trainer.train()
    if cfg.push_to_hub:
        trainer.push_to_hub()
