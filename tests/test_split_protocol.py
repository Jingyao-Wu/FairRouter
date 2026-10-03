"""Shared evaluation identity and prepared-input loading checks."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import torch

from eargtc.feature_transforms.core import Transform
from eargtc.perturbations.assignment import mixed_assignment
from fairrouter.artifacts import Bundle, check_split, sha256
from fairrouter.compatibility import (
    canonical_split, input_modules, input_specification, load_estimator,
)
from scripts import make_splits

ROOT = Path(__file__).resolve().parents[1]


class SplitProtocolTests(unittest.TestCase):
    def test_repository_splits_and_checksums(self):
        if not (ROOT / "splits/3/42/cora.json").is_file():
            self.skipTest("Generate the repository splits before checking their checksums")
        manifest = json.loads((ROOT / 'configs/bundle_manifest.json').read_text())
        hashes = dict(line.split('  ', 1)[::-1] for line in
                      (ROOT / 'splits/SHA256SUMS').read_text().splitlines())
        shared = {}
        self.assertEqual(len(manifest['cells']), 45)
        self.assertEqual(len(hashes), 45)
        for cell in manifest['cells']:
            path = ROOT / cell['split']
            split = json.loads(path.read_text())
            check_split(split, cell)
            self.assertEqual(split, canonical_split(split))
            self.assertEqual(sha256(path), cell['split_sha256'])
            self.assertEqual(sha256(path), hashes[str(path.relative_to(ROOT / 'splits'))])
            key = cell['dataset'], cell['seed']
            self.assertEqual(split['standard_eval_ids'],
                             shared.setdefault(key, split['standard_eval_ids']))
        self.assertEqual(len(shared), 15)

    def test_bundle_rejects_reordered_evaluation(self):
        if not (ROOT / "splits/3/42/cora.json").is_file():
            self.skipTest("Generate the repository splits before checking bundle ordering")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = json.loads((ROOT / 'configs/bundle_manifest.json').read_text())
            for cell in manifest['cells']:
                if cell['dataset'] == 'cora' and cell['seed'] == 42:
                    source = ROOT / cell['split']
                    destination = root / cell['split']
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, destination)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            bundle = Bundle(root)
            cell = bundle.cell(3, 42, 'cora')
            split = bundle.split(cell)
            changed = deepcopy(split)
            changed['standard_eval_ids'].reverse()
            path = root / cell['split']
            path.write_text(json.dumps(changed))
            for entry in manifest['cells']:
                if entry['shot'] == 3 and entry['seed'] == 42 and entry['dataset'] == 'cora':
                    entry['split_sha256'] = sha256(path)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            bundle = Bundle(root)
            with self.assertRaisesRegex(ValueError, 'order must match'):
                bundle.split(bundle.cell(3, 42, 'cora'))

    def test_generator_shares_ordered_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                device=torch.device('cpu'), datasets=['cora'], seeds=[42],
                shots=[3, 5, 10], data_root=Path(directory),
                arxiv_split_dir=None, output=Path(directory) / 'splits',
            )
            def partition(graph, dataset, shots, seed, device, official):
                return dict(
                    dataset=dataset, shots=shots, split_seed=seed, num_nodes=1100,
                    graph_sha256='fixture', support_ids=list(range(shots)),
                    valid_ids=list(range(10, 15)),
                    unlabeled_ids=list(range(shots, 10)) + list(range(15, 1100)),
                    test_truth_exported=False,
                )
            with patch.object(make_splits, 'parse_args', return_value=args), \
                 patch.object(make_splits.torch.cuda, 'is_available', return_value=True), \
                 patch.object(make_splits, 'load_graph', return_value=object()), \
                 patch.object(make_splits, 'sample_partition', side_effect=partition):
                make_splits.main()
            splits = [json.loads((args.output / f'{shot}/42/cora.json').read_text())
                      for shot in args.shots]
            for split in splits:
                self.assertEqual(split, canonical_split(split))
                self.assertEqual(split['standard_eval_ids'], splits[-1]['standard_eval_ids'])
                self.assertTrue(set(split['standard_eval_ids']) <= set(split['unlabeled_ids']))

    def test_assignment_stays_deterministic(self):
        values = mixed_assignment(range(200), seed=303)
        digest = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest, '73a8bd994ed9612f5a8c712ca70e67f7b4ccf3d3af123d55ea1276607b4f42ed')

    def test_prepared_model_class_paths(self):
        x = np.arange(120, dtype=float).reshape(30, 4)
        fitted = Transform.fit(x, np.ones(30))
        expected = fitted.apply(x)
        specification = input_specification()
        source = next(name for name, target in specification['module_names'].items()
                      if target == Transform.__module__)
        source_engine = next(name for name, target in specification['metadata_names'].items()
                             if target == 'tabular')
        with tempfile.TemporaryDirectory() as directory:
            for compression in (0, 3):
                path = Path(directory) / f'model_{compression}.joblib'
                current = Transform.__module__
                with input_modules():
                    try:
                        Transform.__module__ = source
                        joblib.dump(dict(engine=source_engine, transform=fitted),
                                    path, compress=compression)
                    finally:
                        Transform.__module__ = current
                restored = load_estimator(path)
                self.assertEqual(restored['engine'], 'tabular')
                self.assertEqual(type(restored['transform']).__module__, current)
                np.testing.assert_array_equal(restored['transform'].apply(x), expected)


if __name__ == '__main__':
    unittest.main()
