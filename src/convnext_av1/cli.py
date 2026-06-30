"""Command-line interface for ConvNeXt-AV1."""

from __future__ import annotations

import argparse
import json

from .benchmark import consolidate_benchmark
from .converter import convert_partition_log
from .data import create_sequence_split
from .evaluate import evaluate_from_config
from .export import export_checkpoint
from .inspect import inspect_dataset
from .train import train_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="convnext-av1")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("convert", help="Convert libaom binary partition logs to NPZ.")
    p.add_argument("bin_path")
    p.add_argument("output_npz")
    p.add_argument("--manifest")
    p.add_argument("--max-samples", type=int)
    p.add_argument("--export-pickle")

    p = sub.add_parser("inspect", help="Export sample PNGs and a sanity report.")
    p.add_argument("dataset")
    p.add_argument("output_dir")
    p.add_argument("--max-samples", type=int, default=16)

    p = sub.add_parser("split", help="Create sequence-level train/val/test split JSON.")
    p.add_argument("dataset")
    p.add_argument("output_json")
    p.add_argument("--seed", type=int, default=1337)

    p = sub.add_parser("train", help="Train from a YAML config.")
    p.add_argument("config")

    p = sub.add_parser("eval", help="Evaluate a checkpoint from a YAML config.")
    p.add_argument("config")
    p.add_argument("--checkpoint", required=True)

    p = sub.add_parser("export", help="Export a checkpoint to TorchScript or ONNX.")
    p.add_argument("config")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--format", choices=["torchscript", "onnx"], default="torchscript")

    p = sub.add_parser("benchmark", help="Consolidate benchmark CSV into JSON.")
    p.add_argument("input_csv")
    p.add_argument("output_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "convert":
        result = convert_partition_log(args.bin_path, args.output_npz, args.manifest, args.max_samples, args.export_pickle)
    elif args.command == "inspect":
        result = inspect_dataset(args.dataset, args.output_dir, args.max_samples)
    elif args.command == "split":
        result = create_sequence_split(args.dataset, args.output_json, seed=args.seed)
    elif args.command == "train":
        result = train_from_config(args.config)
    elif args.command == "eval":
        result = evaluate_from_config(args.config, args.checkpoint)
    elif args.command == "export":
        result = {"output": export_checkpoint(args.config, args.checkpoint, args.output, args.format)}
    elif args.command == "benchmark":
        result = consolidate_benchmark(args.input_csv, args.output_json)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
