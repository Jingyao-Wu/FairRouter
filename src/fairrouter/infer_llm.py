"""Export frozen LLaMA class logits and final prompt-token representations in shards."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import torch

from .perturbations import delete_span, mask_tokens, mixed_assignment, truncate_text
from .artifacts import read_json, save_tensor, sha256, write_json
from .generation import label_names, validate_split

TEXT_VIEWS = (
    "clean",
    "text_truncate_050",
    "text_mask_mixed_030",
    "text_span_delete_030",
)


def text_view(text, name, node_id):
    if name == "clean":
        return text
    if name == "text_truncate_050":
        return truncate_text(text, 0.5)
    if name == "text_mask_mixed_030":
        if mixed_assignment([node_id], seed=523)[node_id] != "text":
            return text
        return mask_tokens(text, 0.3, seed=523 * 1_000_003 + node_id)
    if name == "text_span_delete_030":
        return delete_span(text, 0.3, seed=877 * 1_000_003 + node_id)
    raise ValueError(f"Unknown text view: {name}")


def prompt(text, profile):
    return f"{text}\n{profile['prompt_suffix']}"


def class_logits(numbered, number_to_label_ids):
    if sorted(number_to_label_ids) != list(range(numbered.shape[1])):
        raise ValueError("Category mapping must be a permutation of label IDs")
    logits = torch.empty_like(numbered, dtype=torch.float32)
    logits[:, number_to_label_ids] = numbered.float()
    return logits


def number_tokens(tokenizer, count):
    pieces = [
        tokenizer.encode(str(i + 1), add_special_tokens=False) for i in range(count)
    ]
    if any(len(p) != 1 for p in pieces) or len({p[0] for p in pieces}) != count:
        raise ValueError(
            "Each category number must correspond to a distinct single token"
        )
    return [p[0] for p in pieces]


@torch.inference_mode()
def answer_outputs(model, encoded, token_ids):
    """Select the final prompt state and class logits from full-sequence projection."""
    mask = encoded["attention_mask"].bool()
    if mask.ndim != 2 or not mask.any(1).all():
        raise ValueError("Every prompt must have at least one attended token")
    positions = (
        torch.arange(mask.shape[1], device=mask.device)
        .expand_as(mask)
        .masked_fill(~mask, -1)
        .max(1)
        .values
    )
    result = model.model(**encoded, use_cache=False, return_dict=True)
    hidden = result.last_hidden_state
    answer = hidden[
        torch.arange(len(mask), device=hidden.device), positions.to(hidden.device)
    ]
    # Preserve the full-sequence vocabulary projection: changing its matrix shape
    # can change FP16 rounding. Calling the module preserves offload hooks.
    head = model.get_output_embeddings()
    head_input = (
        hidden if head.weight.device.type == "meta" else hidden.to(head.weight.device)
    )
    all_logits = head(head_input)
    rows = torch.arange(len(mask), device=all_logits.device)
    vocab_logits = all_logits[rows, positions.to(all_logits.device)]
    indices = torch.tensor(token_ids, dtype=torch.long, device=vocab_logits.device)
    logits = vocab_logits.index_select(1, indices)
    return answer.cpu().half(), logits.float().cpu()


def model_identity(source):
    location = Path(source)
    if not location.is_dir():
        raise ValueError("Provide a local model directory with weights and tokenizer")
    files = {
        p.relative_to(location).as_posix(): sha256(p)
        for p in sorted(location.rglob("*"))
        if p.is_file()
        and p.suffix in (".safetensors", ".bin", ".json", ".model", ".txt", ".tiktoken")
    }
    if not any(name.endswith((".safetensors", ".bin")) for name in files):
        raise ValueError("Model directory contains no weight files")
    return dict(files_sha256=files)


def generate(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    profile = read_json(args.profile)
    if profile["dataset"] != args.dataset:
        raise ValueError("Prompt profile belongs to another dataset")
    supported = dict(
        use_chat_template=True,
        padding_side="left",
        truncation_side="left",
        position_ids="model_default",
        logit_projection="full_sequence",
    )
    if any(profile.get(key) != value for key, value in supported.items()):
        raise ValueError("Unsupported prompt tokenization or projection settings")
    identity = model_identity(args.model)
    texts = read_json(args.texts)
    if (
        not isinstance(texts, list)
        or not texts
        or any(not isinstance(t, str) for t in texts)
    ):
        raise ValueError(
            "Texts must be a nonempty JSON list indexed by canonical node ID"
        )
    names = label_names(args.labels)
    if names != profile["label_names"]:
        raise ValueError("Class order differs from the prompt profile")
    graph_hash = sha256(args.graph)
    if graph_hash != profile["graph_sha256"]:
        raise ValueError("Graph differs from the prompt profile")
    if (
        args.max_length != profile["max_length"]
        or args.dtype != profile["dtype"]
        or args.batch_size != profile["batch_size"]
    ):
        raise ValueError("Inference settings must match the prompt profile")
    support = set()
    for path in args.splits or []:
        split = read_json(path)
        validate_split(split, len(texts))
        if split["dataset"] != args.dataset or split["graph_sha256"] != graph_hash:
            raise ValueError("Split belongs to another dataset or graph")
        support.update(split["support_ids"])
    augmented_ids = sorted(support) if args.splits else list(range(len(texts)))
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, use_fast=True
    )
    if not tokenizer.chat_template:
        raise ValueError("Use a LLaMA instruction model with a chat template")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    tokens = number_tokens(tokenizer, len(names))
    if tokens != profile["number_token_ids"]:
        raise ValueError("Tokenizer category IDs differ from the prompt profile")
    dtype = getattr(torch, args.dtype)
    kwargs = dict(torch_dtype=dtype, local_files_only=True)
    if args.device == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.device != "auto":
        model.to(args.device)
    model.eval()
    if model.config.model_type != "llama" or model.config.hidden_size != 4096:
        raise ValueError(
            "FairRouter requires a LLaMA model with 4096-dimensional hidden states"
        )
    if args.max_length > model.config.max_position_embeddings:
        raise ValueError("max-length exceeds this model's context size")
    empty_prompt = tokenizer.apply_chat_template(
        [dict(role="user", content=prompt("", profile))],
        tokenize=False,
        add_generation_prompt=True,
    )
    if (
        len(
            tokenizer.encode(
                empty_prompt, add_special_tokens=profile["add_special_tokens"]
            )
        )
        >= args.max_length
    ):
        raise ValueError(
            "max-length must leave room for text after the category instructions"
        )
    request = dict(
        schema=2,
        dataset=args.dataset,
        graph_sha256=graph_hash,
        texts_sha256=sha256(args.texts),
        label_names=names,
        num_nodes=len(texts),
        model=identity,
        profile=profile,
        profile_sha256=sha256(args.profile),
        dtype=args.dtype,
        runtime=dict(
            torch=str(torch.__version__),
            transformers=version("transformers"),
            accelerate=version("accelerate"),
            cuda=torch.version.cuda,
            cuda_devices=[
                dict(
                    name=torch.cuda.get_device_name(i),
                    capability=list(torch.cuda.get_device_capability(i)),
                )
                for i in range(torch.cuda.device_count())
            ],
            model_device_map={
                key: str(value)
                for key, value in getattr(model, "hf_device_map", {}).items()
            },
            input_device=str(model.get_input_embeddings().weight.device),
            allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            attention_implementation=getattr(
                model.config, "_attn_implementation", None
            ),
        ),
        max_length=args.max_length,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        number_token_ids=tokens,
        augmented_node_ids=augmented_ids,
        prompt_template=prompt("{text}", profile),
        chat_template=tokenizer.chat_template,
        hidden_semantics="final attended prompt token",
        logit_projection="full sequence vocabulary projection",
        augmentation_seeds=dict(mask=523, span_delete=877),
        node_seed_multiplier=1_000_003,
        package_parity_verified=False,
    )
    fingerprint = hashlib.sha256(
        json.dumps(request, sort_keys=True).encode()
    ).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "request.json"
    if config_path.exists():
        if read_json(config_path) != json.loads(json.dumps(request)):
            raise ValueError(
                "Existing LLM cache uses different inputs or inference settings"
            )
    else:
        if any(args.output.iterdir()):
            raise ValueError("Output directory is not an LLM cache")
        write_json(config_path, request)
    manifest = dict(request, request_sha256=fingerprint, views={})
    completed_path = args.output / "manifest.json"
    completed = read_json(completed_path) if completed_path.exists() else None
    completed_hashes = (
        {r["path"]: r["sha256"] for view in completed["views"].values() for r in view}
        if completed
        else {}
    )
    device = model.get_input_embeddings().weight.device
    for view in TEXT_VIEWS:
        ids = list(range(len(texts))) if view == "clean" else augmented_ids
        shards = []
        for offset in range(0, len(ids), args.shard_size):
            node_ids = ids[offset : offset + args.shard_size]
            relative = f"{view}/{offset:09d}.pt"
            path = args.output / relative
            checksum = path.with_suffix(".sha256")
            if path.exists():
                current = sha256(path)
                if not checksum.exists() or checksum.read_text().strip() != current:
                    raise ValueError("LLM shard checksum differs")
                if completed and completed_hashes.get(relative) != current:
                    raise ValueError("LLM shard differs from the completed manifest")
                cached = torch.load(path, map_location="cpu", weights_only=True)
                validate_shard(cached, node_ids, len(names), fingerprint)
            else:
                hidden_rows, logit_rows = [], []
                for start in range(0, len(node_ids), args.batch_size):
                    batch_ids = node_ids[start : start + args.batch_size]
                    rendered = [
                        tokenizer.apply_chat_template(
                            [
                                dict(
                                    role="user",
                                    content=prompt(
                                        text_view(texts[i], view, i), profile
                                    ),
                                )
                            ],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        for i in batch_ids
                    ]
                    encoded = tokenizer(
                        rendered,
                        padding=True,
                        truncation=True,
                        max_length=args.max_length,
                        add_special_tokens=profile["add_special_tokens"],
                        return_tensors="pt",
                    )
                    encoded = {
                        k: v.to(device)
                        for k, v in encoded.items()
                        if k in ("input_ids", "attention_mask")
                    }
                    hidden, logits = answer_outputs(model, encoded, tokens)
                    hidden_rows.append(hidden)
                    logit_rows.append(
                        class_logits(logits, profile["number_to_label_ids"])
                    )
                cached = dict(
                    node_ids=torch.tensor(node_ids),
                    hidden=torch.cat(hidden_rows),
                    logits=torch.cat(logit_rows),
                    changed_mask=torch.tensor(
                        [text_view(texts[i], view, i) != texts[i] for i in node_ids]
                    ),
                    request_sha256=fingerprint,
                )
                validate_shard(cached, node_ids, len(names), fingerprint)
                # An interrupted write never becomes a completed shard.
                temporary = path.with_suffix(".tmp")
                if temporary.exists():
                    temporary.unlink()
                save_tensor(temporary, cached)
                temporary.replace(path)
                checksum.write_text(sha256(path) + "\n")
            shards.append(dict(path=relative, sha256=sha256(path), rows=len(node_ids)))
            print(
                f"LLM {view}: {min(offset + args.shard_size, len(ids))}/{len(ids)}",
                flush=True,
            )
        manifest["views"][view] = shards
    path = args.output / "manifest.json"
    if path.exists():
        if read_json(path) != json.loads(json.dumps(manifest)):
            raise ValueError("Completed cache manifest differs")
    else:
        write_json(path, manifest)
    print(f"Generated LLaMA outputs: {args.output}", flush=True)


def validate_shard(value, ids, classes, fingerprint):
    if value["request_sha256"] != fingerprint or not torch.equal(
        value["node_ids"], torch.tensor(ids)
    ):
        raise ValueError("LLM shard input identity differs")
    if value["hidden"].shape != (len(ids), 4096) or value["logits"].shape != (
        len(ids),
        classes,
    ):
        raise ValueError("LLM shard dimensions differ")
    if (
        not torch.isfinite(value["hidden"]).all()
        or not torch.isfinite(value["logits"]).all()
    ):
        raise ValueError("Nonfinite LLM outputs")
    if (
        value["changed_mask"].shape != (len(ids),)
        or value["changed_mask"].dtype != torch.bool
    ):
        raise ValueError("Invalid text change mask")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        required=True,
        help="Graph file used to identify canonical node order",
    )
    parser.add_argument(
        "--texts", type=Path, required=True, help="JSON list, one text per graph node"
    )
    parser.add_argument(
        "--labels", type=Path, required=True, help="JSON list in class-ID order"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--model",
        required=True,
        help="Local LLaMA3-8B-Instruct weights and tokenizer directory",
    )
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        type=Path,
        help="Restrict augmented inference to the union of support IDs",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="auto", help="auto for Accelerate placement, or e.g. cuda:0"
    )
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shard-size", type=int, default=512)
    args = parser.parse_args()
    if min(args.max_length, args.batch_size, args.shard_size) < 1:
        parser.error("Lengths and batch sizes must be positive")
    generate(args)


if __name__ == "__main__":
    main()
