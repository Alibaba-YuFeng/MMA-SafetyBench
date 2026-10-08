# MMA-SafetyBench

**A Benchmark for Multimodal Agent Safety Evaluation**

[Dataset](https://huggingface.co/datasets/Alibaba-YuFeng/MMA-SafetyBench)
 · [Evaluation notes](docs/EVALUATION.md)
 · [Code review & remaining gaps](docs/CODE_REVIEW.md)

MMA-SafetyBench evaluates how multimodal agents respond to adversarial visual
content across web, desktop GUI, mobile, document, and video tasks. The benchmark
contains **1,083 adversarial trajectories**. These scripts evaluate model outputs;
a predicted click or target-matching answer is not evidence that an action was
executed in a live environment.

## Coverage

| Domain | Tasks | Local entry point | Status |
| --- | ---: | --- | --- |
| Web automation | 200 | `mma-bench web` | Base and defense adapters |
| Desktop GUI | 209 | `mma-bench gui` | Base and defense use coordinate-target ASR |
| Mobile navigation | 283 | `mma-bench mobile` | Base and defense adapters |
| Documents: receipts/invoices | 100 | `mma-bench invoice` | Base and defense adapters |
| Documents: resumes | 100 | `mma-bench resume` | Base and defense adapters |
| Video | 191 | Legacy scripts under `05_Video_Agent/` | Original renderer missing upstream |

The code cleanup includes offline tests and asset preflight, not a re-run of the
paper's experiments. See [known protocol differences](docs/CODE_REVIEW.md) before
comparing outputs from different scripts.

## Quick start

Python 3.10 or newer is required. Install from a cloned checkout (editable mode
is intentional: the adapters are retained in their original directories).

```bash
git clone https://github.com/Alibaba-YuFeng/MMA-SafetyBench.git
cd MMA-SafetyBench
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Download and extract the dataset separately. Point `MMA_DATA_ROOT` at the directory
containing the five numbered domain folders, not at an individual image folder.
Dataset access is governed by the Hugging Face repository's current permissions.

```text
/path/to/MMA-SafetyBench-data/
├── 01_Web_GUI_Testing/
├── 02_Document_Analysis/
├── 03_web/
├── 04_Mobile_Navigation/
└── 05_Video_Agent/
```

Validate metadata, counts, and image decoding **without API calls or credentials**:

```bash
mma-bench gui --data-root /path/to/MMA-SafetyBench-data --dry-run
mma-bench web --data-root /path/to/MMA-SafetyBench-data --dry-run
```

Configure your OpenAI-compatible provider in environment variables. URLs may be
base URLs ending in `/v1` or full `/chat/completions` URLs. Never commit real keys.
The `.env.example` file documents configuration; it is not loaded automatically.

```bash
export MMA_DATA_ROOT=/path/to/MMA-SafetyBench-data
export VICTIM_API_URL=https://your-provider.example/v1
export VICTIM_API_KEY='your-private-key'
export VICTIM_MODEL='your-model-id'
export JUDGE_API_URL=https://your-judge-provider.example/v1
export JUDGE_API_KEY='your-private-key'
export JUDGE_MODEL='gpt-5'

# Small paid smoke test: requires configured providers; not a full benchmark result.
mma-bench gui --limit 2 --output results/gui-smoke

# Full runs; use a different empty output directory for each model/configuration.
mma-bench web --output results/web-base
mma-bench web --defense --output results/web-defense
```

Only run API evaluation when you intend to send benchmark images and prompts to
the selected providers and incur their usage charges. Attack strings are test
data, not instructions to execute against external systems.

## Results and error handling

Each run writes task-level JSON, a credential-free `config.json`, and an updated
`summary.json`. Document and Web adapters also retain their detailed records in
`raw/`. Existing output directories must be empty to prevent accidental reuse of
results from a different model or configuration.

- The expected sample count does not shrink when an image, API call, or judge fails.
- Failed tasks have `status: "error"` and null scores, not a fabricated score of 1.
- Final `asr_percent` and `scr_percent` remain null until every task has a valid result.
- Partial runs report `*_lower_bound_percent`, successes divided by the full
  expected count, alongside errors and pending tasks. Do not report these as final ASR/SCR.
- GUI always retains a denominator of **209**, including smoke tests.

For coordinate conventions, score definitions, and recalculating historical GUI
defense results, read [Evaluation notes](docs/EVALUATION.md).

## Repository layout

```text
mma_safetybench/       Shared CLI, API validation, asset resolution, and metrics
Benchmark_Dataset/    Original per-domain evaluators and metadata
tests/                Offline regression tests; no paid model requests
docs/                 Evaluation protocol and audit notes
.env.example          Configuration names, without credentials
```

## Development

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
ruff check .
ruff format --check .
```

Do not change prompts, score rules, image preprocessing, or task selection merely
to improve a headline metric. Record such changes as a new protocol and re-run
the affected experiments. Existing historical paper results are not modified by
this software cleanup.

## License and release notes

The upstream snapshot reviewed for this cleanup contains no repository-wide
`LICENSE`. Maintainers must choose and approve a code license before claiming
that the code is licensed for unrestricted reuse. Dataset and source-asset
licenses are separate; consult the dataset documentation. No license is inferred
from public hosting.
