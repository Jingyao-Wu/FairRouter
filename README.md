# FairRouter

Code for **Whether to Trust GNNs or LLMs? FairRouter for Few-Shot Node Classification**.

[Paper](paper/Whether_to_Trust_GNNs_or_LLMs.pdf) · [Method figure](paper/FairRouter_framework.pdf)

## Overview

FairRouter learns which modality to trust for few-shot node classification on text-attributed graphs. It combines a frozen GNN and a frozen LLM through two stages:

1. **Quality-Aware Fair Routing.** Probability, structural and gold-prototype evidence describe each node. On agreement nodes, the router scores the shared prediction. On disagreement nodes, a trust score estimates whether either prediction is correct, and a preference score chooses between the GNN and LLM. Branch-specific ranking selects pseudo-labels. Semantic and structural augmentations provide additional routing observations from the labeled support set.
2. **Multi-modal Residual Classifier.** A residual MLP combines graph and language representations. Its output is fused with the two expert distributions using learned positive weights. Training uses gold labels, separately averaged GNN- and LLM-selected pseudo-label losses, and an optional agreement-preservation loss. The classifier predicts every query node, including nodes excluded from pseudo-label supervision.

[![FairRouter framework: quality-aware fair routing and multimodal residual classification](paper/FairRouter_framework.png)](paper/FairRouter_framework.pdf)

The fused distribution is

$$
p_i^J = \frac{\alpha p_i^G + \beta p_i^L + p_i^R}{\alpha + \beta + 1},
\qquad \alpha,\beta > 0.
$$

The GNN and LLM remain fixed during routing and residual training. See Section 3 and Appendices A–C of the paper for the method.

## Installation

Use Python 3.11. The dependencies include PyTorch 2.6.0 with CUDA 12.4; exact versions are in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

Router and residual-classifier training run on CPU using prepared expert outputs and representations.

## Data preparation

The main experiments use **Cora, Citeseer, PubMed, ogbn-arxiv and the ogbn-products subset**, with **3, 5 and 10 labels per class** and seeds **42, 43 and 44**. The default experts in the paper are a GCN and LLaMA3-8B.

FairRouter consumes frozen expert probabilities and representations, in addition to graph data and few-shot splits. The prepared data/model package is the authenticated input to the benchmark runner. The expert reconstruction commands below are verified separately against that package. To use the prepared package, extract it with:

```bash
mkdir -p data
tar -xf /path/to/fairrouter-data.tar -C data
```

This should create:

```text
data/
├── frozen/      Expert outputs, representations and model parameters
└── frontend/    Graph inputs and augmented expert representations
```

**The prepared-package download link is not yet available.** Release information is recorded in [configs/distribution.json](configs/distribution.json). This approximately 24 GB package contains GNN class probabilities and 128-dimensional representations, LLaMA3-8B class probabilities and 4096-dimensional representations, augmented expert outputs, and model parameters. Expert reconstruction also needs the preprocessed graphs, node texts, class names, and local LLaMA3-8B-Instruct weights. Regenerated tensors are not automatically interchangeable with the package.

## Generate data splits

Install the graph-loading dependency:

```bash
python -m pip install -e '.[splits]'
```

Place the preprocessed PyG files `cora.pt`, `citeseer.pt`, `pubmed.pt`, `arxiv.pt`
and `ogbn-products.pt` in `data/graphs/`. For Arxiv, also provide the official OGB
time-split files `train.csv.gz`, `valid.csv.gz` and `test.csv.gz` in
`data/ogbn_arxiv/split/time/`.

Generate the splits on a CUDA GPU:

```bash
python scripts/make_splits.py \
  --data-root data/graphs \
  --arxiv-split-dir data/ogbn_arxiv/split/time \
  --datasets cora citeseer pubmed arxiv ogbn-products \
  --shots 3 5 10 --seeds 42 43 44 --device cuda:0 \
  --output splits
```

This generates `splits/<shot>/<seed>/<dataset>.json` and `splits/SHA256SUMS`.
The JSON files are ignored by Git; the checksum manifest is retained. Verify it
from the split directory with `sha256sum -c SHA256SUMS`.
`evaluation_1000_ids` contains each shot's own query sample; `standard_eval_ids`
contains the shared 10-shot evaluation set. Generating these files does not
replace the prepared package's authenticated split files or manifest.

The split protocol is unchanged:

- **Support:** sample up to the requested number of nodes per class from the
  input training mask; Arxiv uses the official training split.
- **Validation:** Arxiv uses its official validation set. The other datasets,
  including Products, sample up to 500 nodes outside the input training mask.
- **Query:** all nodes outside support and validation. For Arxiv, this comprises
  the official test nodes and the unused official training nodes.
- **Evaluation:** sample 1,000 query nodes from the 10-shot partition for each
  dataset and seed, and reuse those IDs in the same order across all label budgets.

Keep graph nodes, texts and splits in the same node order. CUDA sampling and the supplied seeds are required to reproduce the supplied split protocol.

## Reconstruct expert inputs

Install `python -m pip install -e '.[experts]'`. Provide trusted PyG graph files,
a JSON list of node texts in graph node order, and a JSON list of class names in
numeric label order. Use the dataset-specific recipes in `configs/experts/` and
prompt profiles in `configs/llm/`; the model directory must contain local
LLaMA3-8B-Instruct weights and tokenizer files.

### GCN encoder and classification head

```bash
python -m fairrouter.pretrain_gnn \
  --graph data/graphs/cora.pt --split splits/3/42/cora.json \
  --labels data/labels/cora.json --recipe configs/experts/cora.json \
  --encoder-checkpoint data/encoders/cora.pt \
  --output data/generated/gnn/3/42/cora --device cuda:0
```

`--encoder-checkpoint` accepts the hash-bound encoder-and-embeddings checkpoint
specified by the recipe. Omit it to pretrain with the explicit dataset recipe.
The objective contains masked-feature reconstruction and edge prediction;
Cora, Citeseer, PubMed and Products also use view alignment and variance losses.
The encoder seed and representations are shared across label budgets and split
seeds. Head initialization seeds, class weights, widths and selection rules come
from the per-cell recipe. The five 5-shot/seed42 cells require their fixed
`--head-checkpoint`; the remaining cells refit the head.

The raw graph container includes labels. Only support and validation labels are
retained for head fitting and checkpoint selection; query labels are not used
for training. This distinction is recorded in the output metadata.

### LLM inference

```bash
python -m fairrouter.infer_llm \
  --graph data/graphs/cora.pt --texts data/texts/cora.json \
  --labels data/labels/cora.json --dataset cora \
  --profile configs/llm/cora.json --model data/models/Llama-3-8B-Instruct \
  --splits splits/3/42/cora.json --output data/generated/llm/cora --device auto
```

Profiles specify the exact prompt suffix, category-token IDs, mapping back to
class IDs, special-token handling and input length. Arxiv uses a nonidentity
category-number mapping. Inference uses FP16, one prompt per batch and the final
attended token's hidden state. Class logits retain the full-sequence vocabulary
projection rather than changing the FP16 matrix multiplication shape. These
settings are checked against the profile.

Text views use 50% truncation, 30% token masking and 30% span deletion. Masking
uses assignment seed 523; deletion uses seed 877; node seeds are
`environment_seed * 1000003 + node_id`. Graph views use undirected-pair dropout
with seed 101 and mixed ego-edge masking with seed 303. The clean LLM view covers
all nodes. `--splits` restricts text augmentations to the union of their support
nodes, so pass every split required for subsequent verification.

Cache identity includes content hashes of model and tokenizer files, the prompt
profile, library versions, GPU types and inference settings. Each completed
shard has a checksum; changed
shards are rejected on resume. New expert outputs are marked unverified until
compared with the prepared package.

### Verify generated experts and use the benchmark runner

```bash
python -m fairrouter.generated verify \
  --bundle data/frozen --frontend data/frontend \
  --gnn data/generated/gnn/3/42/cora --llm data/generated/llm/cora \
  --output outputs/cora_expert_verification.json

python -m fairrouter.generated run \
  --bundle data/frozen --frontend data/frontend \
  --gnn data/generated/gnn/3/42/cora --llm data/generated/llm/cora \
  --output outputs/main/cells --mode refit-all
```

Verification requires identical split IDs, clean expert tensors, supervision,
graph views and support text views. Differences stop the command. Once verified,
`run` delegates to the same authenticated package runner below: it uses the
selected router estimators, their same-shot/same-seed source datasets, fixed
calibration and acceptance settings, and the existing branch-wise validation
partition. There is no separate logistic-router recipe or random validation
split. This is a compatibility check followed by the package workflow; it is not
a package-free end-to-end training entry point. Newly pretrained encoders or
LLM inference on a different runtime may fail the exact comparison.

## Running with the prepared package

From the repository root, run FairRouter for one dataset and split:

```bash
python -m fairrouter run --bundle data/frozen --output outputs/example \
  --dataset cora --shot 3 --seed 42 --mode refit-all
```

To train on all five datasets, three label budgets and three seeds:

```bash
for shot in 3 5 10; do
  for seed in 42 43 44; do
    for dataset in cora citeseer pubmed arxiv ogbn-products; do
      python -m fairrouter run --bundle data/frozen --output outputs/main/cells \
        --dataset "$dataset" --shot "$shot" --seed "$seed" --mode refit-all || exit 1
    done
  done
done
```

The residual-classifier hyperparameters are documented in `configs/cells/<shot>/<seed>/<dataset>.json`.

## Evaluation with the prepared package

After training all datasets, label budgets and seeds, run:

```bash
python -m fairrouter evaluate --bundle data/frozen \
  --run outputs/main/cells --output outputs/main/evaluation
```

Results are saved to `outputs/main/evaluation/`:

- `per_seed.csv`: accuracy and Macro-F1 for each dataset, label budget and seed.
- `summary.csv`: mean and standard deviation across seeds.
- `REPORT.md`: an accuracy table for FairRouter and its GCN and zero-shot LLM experts.

Evaluation uses the same 1,000 nodes across label budgets for each dataset and seed. Accuracy summaries use population standard deviation.

## Results reported in the paper

The following values are transcribed from Table 1: node classification accuracy (%, mean ± standard deviation over three seeds).

| Labels per class | Cora | Citeseer | PubMed | Arxiv | Products | Average |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 79.37 ± 0.17 | 73.37 ± 0.12 | 90.80 ± 1.04 | 58.13 ± 0.12 | 72.30 ± 0.85 | 74.79 |
| 5 | 80.70 ± 1.91 | 72.60 ± 0.79 | 91.47 ± 1.30 | 60.40 ± 1.56 | 72.43 ± 0.41 | 75.52 |
| 10 | 81.27 ± 1.36 | 73.47 ± 1.25 | 91.63 ± 0.83 | 61.60 ± 0.86 | 72.80 ± 0.82 | 76.15 |

## Scope

The benchmark runner covers the five datasets, three label budgets and three
seeds using authenticated expert inputs. Expert reconstruction kernels and
explicit recipes are supplied separately; a complete regeneration of all five
datasets' expert caches has not been established as numerically identical.
Raw data/text acquisition, LLaMA pretraining, additional backbones and the full
ablation suite remain outside the supplied workflow.

The evaluator's shared 1,000-node sets should not be equated with the full OGB test splits described in Appendix D.1. The residual trainer implements a shared auxiliary-loss coefficient; the independently weighted auxiliary losses, optional class weighting and fixed-fusion variant described in Appendix C.2 are not exposed by this trainer.

## Code structure

```text
src/fairrouter/   Expert generation, routing, residual training and evaluation
src/eargtc/       Evidence extractors and router estimators
configs/         Per-split hyperparameters and data specifications
scripts/         Data split generation and optional cluster submission
tests/           Protocol, cache-integrity and computation tests
paper/           Paper and method figure
```

## Citation

The paper is under anonymous review. A final citation will be provided when publication details are available.

```bibtex
@misc{fairrouter,
  title  = {Whether to Trust GNNs or LLMs? FairRouter for Few-Shot Node Classification},
  author = {{Anonymous Authors}},
  note   = {Manuscript under review}
}
```

## License

The source code is available under the [MIT License](LICENSE). Datasets and pretrained models remain subject to their respective licenses.
