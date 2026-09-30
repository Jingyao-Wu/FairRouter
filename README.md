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

The supplied commands require Slurm. Router and residual-classifier training run on CPU using prepared expert outputs and representations.

## Data preparation

The main experiments use **Cora, Citeseer, PubMed, ogbn-arxiv and the ogbn-products subset**, with **3, 5 and 10 labels per class** and seeds **42, 43 and 44**. The default experts in the paper are a GCN and LLaMA3-8B.

Data splits are generated locally from the preprocessed graph files using `scripts/make_splits.py`. Expert outputs, representations and model parameters are provided in a separate data/model package. Extract that package first:

```bash
mkdir -p data
tar -xf /path/to/fairrouter-artifacts-v1-anonymous.tar -C data
```

This should create:

```text
data/
├── frozen/      Expert outputs, representations and model parameters
└── frontend/    Graph inputs and augmented expert representations
```

**The data/model download link is not yet available.** Its release information is recorded in `configs/distribution.json`. The source code alone is insufficient to run the experiments. The package must include both directories above; raw benchmark downloads alone do not provide the required expert representations.

## Generate data splits

Install the graph-loading dependency:

```bash
python -m pip install -e '.[splits]'
```

Place the preprocessed PyG files `cora.pt`, `citeseer.pt`, `pubmed.pt`, `arxiv.pt`
and `ogbn-products.pt` in `data/graphs/`. For Arxiv, also provide the official OGB
time-split files `train.csv.gz`, `valid.csv.gz` and `test.csv.gz` in
`data/ogbn_arxiv/split/time/`.

Generate the splits on a CUDA GPU through Slurm:

```bash
srun --partition="<gpu-partition>" --ntasks=1 --gres=gpu:1 --cpus-per-task=2 --mem=24G \
  python scripts/make_splits.py \
  --data-root data/graphs \
  --arxiv-split-dir data/ogbn_arxiv/split/time \
  --datasets cora citeseer pubmed arxiv ogbn-products \
  --shots 3 5 10 --seeds 42 43 44 --device cuda:0 \
  --output data/frozen/splits
```

This generates `data/frozen/splits/<shot>/<seed>/<dataset>.json`, the location
used by the training command below. Each file contains the support, validation,
query and evaluation node IDs. No separate download of split JSON files is needed.

The split protocol is unchanged:

- **Support:** sample up to the requested number of nodes per class from the
  input training mask; Arxiv uses the official training split.
- **Validation:** Arxiv uses its official validation set. The other datasets,
  including Products, sample up to 500 nodes outside the input training mask.
- **Query:** all nodes outside support and validation. For Arxiv, this comprises
  the official test nodes and the unused official training nodes.
- **Evaluation:** sample 1,000 query nodes from the 10-shot partition for each
  dataset and seed, and reuse those IDs in the same order across all label budgets.

`standard_eval_ids` stores the shared evaluation set used by the evaluator;
`evaluation_1000_ids` stores the query sample for the individual label budget.
The script also computes the 10-shot partition when only 3 or 5 shots are requested.
CUDA sampling and the supplied seeds are required to reproduce the node IDs.

Use the graph files associated with the model package, preserving their node order
and masks. The runner checks the generated splits against the package's expected
identities. The script leaves identical existing files in place and refuses to
replace different splits; generating JSON files alone does not adapt model inputs
to a different graph or partition.

## Running FairRouter

From the repository root, activate the installed environment and set the partition available on your cluster:

```bash
export FAIRROUTER_PYTHON="$(command -v python)"
export FAIRROUTER_PARTITION="<cpu-partition>"
export FAIRROUTER_FRONTEND="$PWD/data/frontend"
bash run.sh data/frozen outputs/main refit-all
```

This runs the five datasets at all three label budgets and seeds, followed by evaluation. Use a new output directory for each run. The modes are:

| Mode | Operation |
| --- | --- |
| `refit-all` | Train the configured router scorers and residual classifier; reuse the supplied calibration and acceptance settings. |
| `refit-joint` | Use the supplied router and train the residual classifier. |
| `replay` | Predict using the supplied router and residual-classifier parameters. |

For one dataset and split:

```bash
srun --partition="$FAIRROUTER_PARTITION" --ntasks=1 --cpus-per-task=2 --mem=24G \
  python -m fairrouter run --bundle data/frozen --output outputs/cora_3shot \
  --dataset cora --shot 3 --seed 42 --mode refit-joint
```

Dataset arguments are `cora`, `citeseer`, `pubmed`, `arxiv` and `ogbn-products`. Omitting `FAIRROUTER_FRONTEND` from the main command uses the prepared routing features directly.

The per-split residual-classifier settings are listed in `configs/cells/<shot>/<seed>/<dataset>.json`. The runner reads the matching settings from the data package; editing the JSON copies alone does not change a run. In these settings, `dim` is the projection dimension, `cross` is the disagreement-loss weight, `keep` is the agreement-preservation weight, and `aux` is a shared weight for the two auxiliary residual losses.

## Evaluation

The main command writes the following files to `outputs/main/evaluation/`:

- `per_seed.csv`: accuracy and Macro-F1 for each dataset, label budget and seed.
- `summary.csv`: mean and standard deviation across seeds.
- `REPORT.md`: an accuracy table for FairRouter and its GCN and zero-shot LLM experts.

The evaluator uses the generated `standard_eval_ids`: the same ordered set of 1,000 evaluation nodes for every label budget of a dataset and seed. Accuracy summaries use population standard deviation. The expert scores produced here are not the complete baseline comparison in Table 1.

## Results reported in the paper

The following values are transcribed from Table 1: node classification accuracy (%, mean ± standard deviation over three seeds).

| Labels per class | Cora | Citeseer | PubMed | Arxiv | Products | Average |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 79.37 ± 0.17 | 73.37 ± 0.12 | 90.80 ± 1.04 | 58.13 ± 0.12 | 72.30 ± 0.85 | 74.79 |
| 5 | 80.70 ± 1.91 | 72.60 ± 0.79 | 91.47 ± 1.30 | 60.40 ± 1.56 | 72.43 ± 0.41 | 75.52 |
| 10 | 81.27 ± 1.36 | 73.47 ± 1.25 | 91.63 ± 0.83 | 61.60 ± 0.86 | 72.80 ± 0.82 | 76.15 |

## Scope

The supplied entry points cover the five-dataset, three-budget FairRouter experiments using prepared expert inputs. They do not include GNN encoder pretraining, LLM inference, the additional backbones, heterophilous benchmarks, or the full ablation suite.

The evaluator's shared 1,000-node sets should not be equated with the full OGB test splits described in Appendix D.1. The residual trainer implements a shared auxiliary-loss coefficient; the independently weighted auxiliary losses, optional class weighting and fixed-fusion variant described in Appendix C.2 are not exposed by this trainer.

## Code structure

```text
src/fairrouter/   Routing, residual classification, training and evaluation
src/eargtc/       Evidence extractors and router estimators
configs/         Per-split hyperparameters and data specifications
scripts/         Data split generation and experiment submission
paper/           Paper and method figure
run.sh           Main experiment command
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
