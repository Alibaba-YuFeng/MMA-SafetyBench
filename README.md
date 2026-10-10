<div align="center">

# MMA-SafetyBench
### A Benchmark for Multimodal Agent Safety Evaluation

**NeurIPS 2026 · Evaluations and Datasets Track · Poster**

[Yuke Wang](https://openreview.net/profile?id=~Yuke_Wang6)<sup>2,*</sup> · [Benlei Cui](https://openreview.net/profile?id=~Benlei_Cui1)<sup>1,*</sup> · [Shen Pang](https://openreview.net/profile?id=~Shen_Pang1)<sup>2</sup> · [Xuemei Dong](https://openreview.net/profile?id=~Xuemei_Dong1)<sup>2,†</sup> · [Longtao Huang](https://openreview.net/profile?id=~Longtao_Huang2)<sup>1</sup><br>
[Hui Xue](https://openreview.net/profile?id=~Hui_Xue5)<sup>1</sup> · [Yuwen Zhai](https://openreview.net/profile?id=~Yuwen_Zhai1)<sup>1</sup> · [Junjie Li](https://openreview.net/profile?id=~Junjie_Li31)<sup>4</sup> · [Jingqun Tang](https://openreview.net/profile?id=~Jingqun_Tang1)<sup>3</sup> · [Haiwen Hong](https://openreview.net/profile?id=~Haiwen_Hong1)<sup>1</sup>

<sup>1</sup> Yuvion Team, Alibaba Group · <sup>2</sup> Zhejiang Gongshang University<br>
<sup>3</sup> Ant Group · <sup>4</sup> Alibaba Group<br>
<sub>* Equal contribution · † Corresponding author</sub>

[![Dataset](https://img.shields.io/badge/🤗_Hugging_Face-Dataset-FFD21E?style=flat-square)](https://huggingface.co/datasets/Alibaba-YuFeng/MMA-SafetyBench)
[![Evaluation](https://img.shields.io/badge/Protocol-Evaluation_notes-083080?style=flat-square)](docs/EVALUATION.md)
[![Quick start](https://img.shields.io/badge/Get_started-Quick_start-16837B?style=flat-square)](#-quick-start)

**5 domains &nbsp; · &nbsp; 1,083 adversarial trajectories &nbsp; · &nbsp; 9 evaluated models**

</div>

---

## Overview

MMA-SafetyBench evaluates how multimodal agents respond to adversarial visual
content across web, desktop GUI, mobile, document, and video tasks. The benchmark
contains **1,083 adversarial trajectories**. These scripts evaluate model outputs;
a predicted click or target-matching answer is not evidence that an action was
executed in a live environment.

<p align="center">
  <img src="assets/overview.png" alt="MMA-SafetyBench overview: visual injection threats, construction across five domains, and model-output safety evaluation." width="100%">
</p>
<p align="center"><sub>From visual injection to model-output evaluation across five agent domains.</sub></p>

### At a glance

- **Cross-domain coverage:** web automation, desktop GUI, mobile navigation, document analysis, and video tasks.
- **Visual attack scenarios:** adversarial content is embedded in the visual material an agent encounters during a task.
- **Explicit evaluation protocol:** ASR and SCR are documented separately, with task-level outputs and fixed benchmark denominators.

## 📢 News

- **2026-09-25:** MMA-SafetyBench was accepted to the **NeurIPS 2026 Evaluations and Datasets Track** as a poster.

## 🗂️ Benchmark & resources

| Resource | What you will find |
| --- | --- |
| [Hugging Face dataset](https://huggingface.co/datasets/Alibaba-YuFeng/MMA-SafetyBench) | Benchmark assets and dataset documentation; access depends on current repository permissions |
| [Evaluation protocol](docs/EVALUATION.md) | Metrics, coordinate conventions, and historical GUI result compatibility |
| [Implementation notes](docs/CODE_REVIEW.md) | Verified checks, remaining gaps, and protocol differences |
| [Configuration example](.env.example) | Provider configuration without credentials |

### Domain coverage

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

## 🚀 Quick start

### 1. Install

Python 3.10 or newer is required. Install from a cloned checkout (editable mode
is intentional: the adapters are retained in their original directories).

```bash
git clone https://github.com/Alibaba-YuFeng/MMA-SafetyBench.git
cd MMA-SafetyBench
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

### 2. Prepare the data

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

### 3. Check assets offline

Validate metadata, counts, and image decoding **without API calls or credentials**:

```bash
mma-bench gui --data-root /path/to/MMA-SafetyBench-data --dry-run
mma-bench web --data-root /path/to/MMA-SafetyBench-data --dry-run
```

### 4. Configure and evaluate

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
