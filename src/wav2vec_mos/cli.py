import argparse

from dotenv import load_dotenv
from transformers import HfArgumentParser


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(prog="wav2vec-mos", add_help=False)
    parser.add_argument("command", choices=["train", "infer"])
    args, remaining = parser.parse_known_args()

    if args.command == "train":
        from wav2vec_mos.train import Wav2VecConfig, train

        (cfg,) = HfArgumentParser(Wav2VecConfig).parse_args_into_dataclasses(remaining)
        train(cfg)
    else:
        from wav2vec_mos.infer import Wav2VecInferConfig, infer

        (cfg,) = HfArgumentParser(Wav2VecInferConfig).parse_args_into_dataclasses(remaining)
        infer(cfg)
