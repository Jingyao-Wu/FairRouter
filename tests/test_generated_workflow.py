"""Protocol and cache integrity tests for expert reconstruction."""

import argparse
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from fairrouter.artifacts import read_json, sha256
from fairrouter.generation import validate_split
from fairrouter.infer_llm import (
    answer_outputs,
    class_logits,
    generate,
    model_identity,
    prompt,
    text_view,
)
from fairrouter.pretrain_gnn import graph_views


class TinyTokenizer:
    chat_template = "test template"
    pad_token = "pad"
    eos_token = "eos"

    def encode(self, text, add_special_tokens=False):
        return [int(text)] if text.isdigit() else [7, 8, 9]

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def __call__(self, texts, **kwargs):
        rows = [[7, 8 + sum(map(ord, text)) % 20] for text in texts]
        return dict(
            input_ids=torch.tensor(rows),
            attention_mask=torch.ones(len(rows), 2, dtype=torch.long),
        )


class TinyBackbone(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.embedding = nn.Embedding(32, width)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class TinyLlama(nn.Module):
    def __init__(self, width=4096):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="llama", hidden_size=width, max_position_embeddings=4096
        )
        self.model = TinyBackbone(width)
        self.lm_head = nn.Linear(width, 32, bias=False)

    def get_input_embeddings(self):
        return self.model.embedding

    def get_output_embeddings(self):
        return self.lm_head


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_shared_eval_required(self):
        split = dict(
            dataset="cora",
            shots=3,
            split_seed=42,
            num_nodes=1006,
            support_ids=[0, 1],
            valid_ids=[2, 3, 4, 5],
            unlabeled_ids=list(range(6, 1006)),
            test_truth_exported=False,
            standard_eval_ids=list(range(6, 1006)),
        )
        validate_split(split, 1006)
        for bad in ([], list(range(6, 1005)), [0] + list(range(7, 1006))):
            value = dict(split, standard_eval_ids=bad)
            with self.assertRaises(ValueError):
                validate_split(value, 1006)
        del split["standard_eval_ids"]
        with self.assertRaises(KeyError):
            validate_split(split, 1006)

    def test_graph_views_have_canonical_edge_order(self):
        edge = torch.tensor([[2, 0, 1, 1, 2, 3, 0], [1, 1, 2, 2, 2, 0, 3]])
        for _, shifted, _ in graph_views(edge, 4):
            pairs = list(map(tuple, shifted.T.tolist()))
            self.assertEqual(pairs, sorted(set(pairs)))
            self.assertTrue(all(u != v and (v, u) in pairs for u, v in pairs))

    def test_profiles_and_category_permutation(self):
        root = Path(__file__).resolve().parents[1]
        profile = read_json(root / "configs/llm/arxiv.json")
        mapping = profile["number_to_label_ids"]
        self.assertEqual(mapping[:4], [10, 15, 9, 7])
        values = torch.arange(40).reshape(1, 40).float()
        self.assertTrue(torch.equal(class_logits(values, mapping)[:, mapping], values))
        self.assertEqual(prompt("paper", profile), "paper\n" + profile["prompt_suffix"])
        self.assertTrue(profile["add_special_tokens"])
        with self.assertRaises(ValueError):
            class_logits(values, [0] * 40)

    def test_answer_matches_llama_forward(self):
        from transformers import LlamaConfig, LlamaForCausalLM

        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=32,
            )
        ).eval()
        encoded = dict(
            input_ids=torch.tensor([[0, 2, 3], [4, 5, 0]]),
            attention_mask=torch.tensor([[0, 1, 1], [1, 1, 0]]),
        )
        hidden, logits = answer_outputs(model, encoded, [1, 2])
        with torch.no_grad():
            expected = model(**encoded, output_hidden_states=True, use_cache=False)
        rows, positions = torch.tensor([0, 1]), torch.tensor([2, 1])
        torch.testing.assert_close(
            hidden, expected.hidden_states[-1][rows, positions].half()
        )
        torch.testing.assert_close(
            logits, expected.logits[rows, positions][:, [1, 2]], atol=1e-6, rtol=1e-6
        )

    def test_content_identity_ignores_mtime(self):
        folder = self.root / "model"
        folder.mkdir()
        path = folder / "model.safetensors"
        path.write_bytes(b"first")
        before, stat = model_identity(folder), path.stat()
        path.write_bytes(b"other")
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertNotEqual(model_identity(folder), before)

    def test_cache_resume_and_tamper_rejection(self):
        graph, labels, texts, profile = [
            self.root / name
            for name in ("graph.pt", "labels.json", "texts.json", "profile.json")
        ]
        graph.write_bytes(b"graph identity")
        labels.write_text(json.dumps(["alpha", "beta"]))
        texts.write_text(json.dumps(["one two three", "four five six"]))
        profile.write_text(
            json.dumps(
                dict(
                    dataset="cora",
                    graph_sha256=sha256(graph),
                    label_names=["alpha", "beta"],
                    prompt_suffix="choose one",
                    number_token_ids=[1, 2],
                    number_to_label_ids=[1, 0],
                    max_length=128,
                    dtype="float32",
                    batch_size=1,
                    add_special_tokens=True,
                    use_chat_template=True,
                    padding_side="left",
                    truncation_side="left",
                    position_ids="model_default",
                    logit_projection="full_sequence",
                )
            )
        )
        model_dir = self.root / "model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"test weight identity")
        args = argparse.Namespace(
            graph=graph,
            labels=labels,
            texts=texts,
            profile=profile,
            dataset="cora",
            model=model_dir,
            splits=None,
            output=self.root / "cache",
            dtype="float32",
            max_length=128,
            batch_size=1,
            shard_size=2,
            device="cpu",
        )
        model = TinyLlama()
        with (
            patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=TinyTokenizer(),
            ),
            patch(
                "transformers.AutoModelForCausalLM.from_pretrained", return_value=model
            ),
        ):
            generate(args)
            before = sha256(args.output / "manifest.json")
            generate(args)
            self.assertEqual(sha256(args.output / "manifest.json"), before)
            with patch("torch.get_float32_matmul_precision", return_value="changed"):
                with self.assertRaisesRegex(ValueError, "different inputs"):
                    generate(args)
            shard = args.output / "clean/000000000.pt"
            value = torch.load(shard, weights_only=True)
            value["hidden"][0, 0] += 1
            torch.save(value, shard)
            with self.assertRaisesRegex(ValueError, "checksum"):
                generate(args)
            shard.with_suffix(".sha256").write_text(sha256(shard) + "\n")
            with self.assertRaisesRegex(ValueError, "completed manifest"):
                generate(args)

    def test_text_views_are_seeded_by_environment(self):
        from fairrouter.perturbations import delete_span, mask_tokens, mixed_assignment

        text = "one two three four five six seven eight"
        for node in range(20):
            expected = (
                mask_tokens(text, 0.3, seed=523 * 1000003 + node)
                if mixed_assignment([node], seed=523)[node] == "text"
                else text
            )
            self.assertEqual(text_view(text, "text_mask_mixed_030", node), expected)
            self.assertEqual(
                text_view(text, "text_span_delete_030", node),
                delete_span(text, 0.3, seed=877 * 1000003 + node),
            )


if __name__ == "__main__":
    unittest.main()
