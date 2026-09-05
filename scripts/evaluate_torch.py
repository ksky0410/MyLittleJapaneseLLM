"""PyTorch checkpointをdomain別lossとheld-out会話で評価する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from _common import repo_path

from evaluate_chat_dataset import (
    _format_text,
    _read_records,
    _turns,
    encode_history,
    select_examples,
    select_examples_from_manifest,
    summarize_chat_results,
    token_overlap_scores,
)
from train_torch import _evaluation_batches, _generate, _loss
from my_little_japanese_llm.config import load_config
from my_little_japanese_llm.data import load_tokens
from my_little_japanese_llm.tokenizer import load_processor
from my_little_japanese_llm.torch_model import TorchJapaneseGPT, parameter_count, require_torch
from my_little_japanese_llm.training import perplexity, signature_from_config


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device(torch: Any, value: str) -> Any:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    selected = torch.device(value)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cudaが指定されましたがCUDAが利用できません")
    return selected


def _runtime(torch: Any, device: Any, amp_enabled: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "amp_enabled": amp_enabled,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_capability": list(torch.cuda.get_device_capability(device)),
                "gpu_total_memory_bytes": properties.total_memory,
            }
        )
    return result


def _autocast(torch: Any, device: Any, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=enabled,
    )


def _load_model(config: Any, checkpoint_path: Path, device: Any, torch: Any) -> tuple[Any, Any, Any]:
    processor = load_processor(config.paths.tokenizer_model)
    vocab_size = int(processor.vocab_size())
    model = TorchJapaneseGPT(
        vocab_size=vocab_size,
        dim=config.model.dim,
        layers=config.model.layers,
        heads=config.model.heads,
        context_length=config.model.context_length,
        mlp_ratio=config.model.mlp_ratio,
        position_embedding=config.model.position_embedding,
    ).to(device)
    metadata_path = checkpoint_path.with_suffix(".json")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpointが見つかりません: {checkpoint_path}")
    if not metadata_path.is_file():
        raise ValueError(f"checkpoint metadataがありません: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_signature = signature_from_config(config, vocab_size)
    actual_signature = metadata.get("model")
    if isinstance(actual_signature, dict) and "position_embedding" not in actual_signature:
        actual_signature = {**actual_signature, "position_embedding": "absolute"}
    if metadata.get("format_version") != 1 or metadata.get("weights_file") != checkpoint_path.name:
        raise ValueError(f"checkpoint metadataの形式が不正です: {metadata_path}")
    if actual_signature != expected_signature:
        raise ValueError(
            "checkpointと現在の設定が一致しません。"
            f" expected={expected_signature}, actual={actual_signature}"
        )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, processor, metadata


def _evaluate(
    model: Any,
    tokens: Any,
    config: Any,
    device: Any,
    torch: Any,
    functional: Any,
    amp_enabled: bool,
    batches: int,
) -> float:
    losses = []
    with torch.no_grad():
        eval_batches = _evaluation_batches(
            tokens,
            config.training.batch_size,
            config.model.context_length,
            batches,
            device,
            torch,
        )
        for inputs, targets in eval_batches:
            with _autocast(torch, device, amp_enabled):
                losses.append(float(_loss(model, inputs, targets, functional).item()))
    return float(sum(losses) / len(losses))


def evaluate_domains(args: argparse.Namespace) -> dict[str, Any]:
    torch = require_torch()
    import torch.nn.functional as functional

    config = load_config(repo_path(args.config))
    device = _device(torch, args.device)
    amp_enabled = device.type == "cuda" and not args.no_amp
    checkpoint = repo_path(args.checkpoint).resolve()
    model, _, metadata = _load_model(config, checkpoint, device, torch)
    batches = args.eval_batches if args.eval_batches is not None else config.training.eval_batches
    if batches <= 0:
        raise ValueError("--eval-batchesは正の整数で指定してください")
    domains: list[dict[str, Any]] = []
    for name, path_value in args.domain:
        token_path = repo_path(path_value).resolve()
        if not token_path.is_file():
            raise FileNotFoundError(f"Tokenファイルが見つかりません: {token_path}")
        tokens = load_tokens(token_path)
        loss = _evaluate(
            model, tokens, config, device, torch, functional, amp_enabled, batches
        )
        domains.append(
            {
                "name": name,
                "token_path": str(token_path),
                "token_sha256": _sha256_file(token_path),
                "token_count": int(tokens.size),
                "validation_loss": loss,
                "perplexity": perplexity(loss),
            }
        )
    result = {
        "format": "domain-evaluation-torch-v1",
        "backend": "pytorch-cuda" if device.type == "cuda" else "pytorch",
        "config": str(config.source_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "checkpoint_step": metadata.get("metrics", {}).get("step"),
        "parameter_count": parameter_count(model),
        "seed": config.training.seed,
        "eval_batches": batches,
        "runtime": _runtime(torch, device, amp_enabled),
        "domains": domains,
    }
    output = repo_path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def evaluate_chat(args: argparse.Namespace) -> dict[str, Any]:
    torch = require_torch()
    config = load_config(repo_path(args.config))
    device = _device(torch, args.device)
    amp_enabled = device.type == "cuda" and not args.no_amp
    checkpoint = repo_path(args.checkpoint).resolve()
    model, processor, metadata = _load_model(config, checkpoint, device, torch)
    input_path = repo_path(args.input).resolve()
    selection_path = repo_path(args.selection_file).resolve() if args.selection_file else None
    records = _read_records(input_path)
    if selection_path is None:
        examples = select_examples(records, args.examples, args.seed)
    else:
        manifest = json.loads(selection_path.read_text(encoding="utf-8"))
        if manifest.get("input_sha256") != _sha256_file(input_path):
            raise ValueError("評価manifestと入力会話JSONLのSHA-256が一致しません")
        examples = select_examples_from_manifest(records, manifest)
    if not examples:
        raise ValueError("評価可能な2発話以上の会話がありません")

    results: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        turns = example["turns"]
        target_index = int(example["target_index"])
        prompt_ids, rendered_prompt = encode_history(turns, target_index, processor)
        output_ids = _generate(
            model,
            prompt_ids,
            args.max_new_tokens,
            config.model.context_length,
            config.generation.temperature,
            config.generation.top_k,
            args.seed + index,
            int(processor.eos_id()),
            device,
            torch,
        )
        completion_ids = output_ids[len(prompt_ids) :]
        reference_ids = [int(token) for token in processor.encode(turns[target_index]["text"], out_type=int)]
        overlap = token_overlap_scores(reference_ids, completion_ids, int(processor.eos_id()))
        results.append(
            {
                "conversation_id": example["conversation_id"],
                "record_index": example["record_index"],
                "target_index": target_index,
                "target_speaker": turns[target_index]["speaker_id"],
                "source": example.get("source"),
                "stratum": example.get("stratum"),
                "rendered_prompt": rendered_prompt,
                "prompt_token_count": len(prompt_ids),
                "history_truncated": len(prompt_ids) > config.model.context_length,
                "train_text_overlap": example.get("train_text_overlap"),
                "reference": turns[target_index]["text"],
                "reference_token_count": len(reference_ids),
                "completion": processor.decode(completion_ids),
                "generated_token_count": len(completion_ids),
                "eos_reached": int(processor.eos_id()) in completion_ids,
                **overlap,
                "seed": args.seed + index,
            }
        )

    result = {
        "format": "heldout-chat-dataset-evaluation-torch-v1",
        "backend": "pytorch-cuda" if device.type == "cuda" else "pytorch",
        "input": str(input_path),
        "input_sha256": _sha256_file(input_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "checkpoint_step": metadata.get("metrics", {}).get("step"),
        "config": str(config.source_path),
        "seed": args.seed,
        "max_examples": args.examples if selection_path is None else None,
        "selected_example_count": len(results),
        "parameter_count": parameter_count(model),
        "runtime": _runtime(torch, device, amp_enabled),
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": config.generation.temperature,
            "top_k": config.generation.top_k,
        },
        "selection": str(selection_path) if selection_path is not None else None,
        "selection_sha256": _sha256_file(selection_path) if selection_path is not None else None,
        "overall_summary": summarize_chat_results(
            [{**item, "stratum": "overall"} for item in results]
        )["overall"],
        "stratum_summary": summarize_chat_results(results),
        "results": results,
    }
    output = repo_path(args.output).resolve()
    text_output = repo_path(args.text_output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    text_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text_output.write_text(_format_text(result), encoding="utf-8")
    return result


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--no-amp", action="store_true", help="CUDAでもfloat32で評価")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    domains = subparsers.add_parser("domains", help="複数Token列のlossを評価")
    _common_parser(domains)
    domains.add_argument("--domain", action="append", required=True, metavar="NAME=PATH")
    domains.add_argument("--eval-batches", type=int, default=None)
    domains.add_argument("--output", required=True)
    domains.set_defaults(handler=evaluate_domains)

    chat = subparsers.add_parser("chat", help="held-out会話を生成評価")
    _common_parser(chat)
    chat.add_argument("--input", required=True)
    chat.add_argument("--output", required=True)
    chat.add_argument("--text-output", required=True)
    chat.add_argument("--examples", type=int, default=48)
    chat.add_argument("--max-new-tokens", type=int, default=64)
    chat.add_argument("--seed", type=int, default=42)
    chat.add_argument("--selection-file", default=None)
    chat.set_defaults(handler=evaluate_chat)

    args = parser.parse_args()
    if args.command == "domains":
        parsed_domains = []
        for value in args.domain:
            name, separator, path = value.partition("=")
            if not separator or not name.strip() or not path.strip():
                parser.error("--domainはNAME=PATH形式で指定してください")
            parsed_domains.append((name.strip(), path.strip()))
        args.domain = parsed_domains
    result = args.handler(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
