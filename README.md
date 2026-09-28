<div align="center">

# MMA-SafetyBench
### A Benchmark for Multimodal Agent Safety Evaluation

**NeurIPS 2026 · Evaluations & Datasets Track · Poster**

[![Dataset](https://img.shields.io/badge/🤗%20Dataset-Hugging%20Face-FFD21E)](https://huggingface.co/datasets/Alibaba-YuFeng/MMA-SafetyBench)
[![Code](https://img.shields.io/badge/Code-GitHub-181717?logo=github)](https://github.com/Alibaba-YuFeng/MMA-SafetyBench)

**Web · Document · GUI · Mobile · Video**

[Overview](#overview) · [Dataset](#dataset) · [Getting started](#getting-started) · [Evaluation guide](docs/EVALUATION.md) · [Responsible use](#responsible-use)

</div>

## Overview

MMA-SafetyBench evaluates how multimodal agents respond to malicious instructions embedded in the content they observe. It covers five domains, from web and mobile interfaces to documents and video, to study whether agents follow the legitimate task or are redirected by injected content.

The benchmark distinguishes **attack success** from **semantic compromise**. For coordinate-based evaluations, a predicted click on an attack target is an output-level success signal—not evidence that an action was executed in a live environment.

## Dataset

Download the data from [Hugging Face](https://huggingface.co/datasets/Alibaba-YuFeng/MMA-SafetyBench). This GitHub repository contains evaluation scripts and selected metadata; cloning it alone does not provide all image and video assets.

| Domain | Attack-evaluation samples¹ | Code directory |
| :--- | ---: | :--- |
| Web | 200 | [03_web](Benchmark_Dataset/03_web) |
| Document | 200 | [02_Document_Analysis](Benchmark_Dataset/02_Document_Analysis) |
| GUI | 209 | [01_Web_GUI_Testing](Benchmark_Dataset/01_Web_GUI_Testing) |
| Mobile | 283 | [04_Mobile_Navigation](Benchmark_Dataset/04_Mobile_Navigation) |
| Video | 191 | [05_Video_Agent](Benchmark_Dataset/05_Video_Agent) |
| **Total** | **1,083** | |

¹ These are the paper's attack-evaluation denominators, not counts independently verified from the downloadable release. Clean-task evaluation uses a separate sample set; in particular, its GUI denominator is 42, not 209.

## Getting started

### 1. Get the code

```bash
git clone https://github.com/Alibaba-YuFeng/MMA-SafetyBench.git
cd MMA-SafetyBench
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The dependency list is inferred from the current source imports, not a locked environment from the original experiments. Video scripts have additional dependencies listed in `requirements-video.txt` and require additional assets described in the [evaluation guide](docs/EVALUATION.md).

### 2. Prepare data and configuration

Download the relevant data assets, then configure the chosen script's model identifiers, API endpoints, credentials, input paths, and output directory. The current runners use in-file configuration; they do not share a unified command-line interface. Keep credentials out of commits and use a separate output directory for each model and prompt condition.

### 3. Select a runner

See the [evaluation guide](docs/EVALUATION.md) for the exact entry points and required directories. Do not launch a full run until a small sample has passed the path, response-format, and scoring checks. Model and judge calls can incur API charges.

## Evaluation metrics

- **ASR — Attack Success Rate:** the proportion of evaluated samples that satisfy the specified attack-success criterion. The criterion and denominator must be recorded for each run.
- **SCR — Semantic Compromise Rate:** the proportion assigned score 3 or 5 by the semantic judge. SCR is **not** clean-task success rate.
- **Clean-task success:** performance on legitimate tasks without the attack; report it separately from ASR and SCR.

Preserve per-sample outputs, failure records, prompt conditions, and judge scores when reporting results. The current GUI runners use different success rules; read the [implementation notes](docs/EVALUATION.md#implementation-notes) before comparing their outputs.

## Repository layout

```text
Benchmark_Dataset/
├── 01_Web_GUI_Testing/     # GUI evaluation and popup-generation scripts
├── 02_Document_Analysis/   # Document and resume evaluation
├── 03_web/                # Web evaluation and metadata
├── 04_Mobile_Navigation/  # Mobile evaluation
└── 05_Video_Agent/        # Video agent, tools, and evaluation
docs/
└── EVALUATION.md          # Runner map, configuration, and implementation notes
```

The existing directory layout is retained so that this documentation update does not alter experiment paths or behavior.

## Responsible use

Use these materials for authorized safety research in isolated environments. Treat injected instructions as untrusted benchmark content: do not follow embedded links, submit real credentials, or execute actions against third-party systems. Review assets for personal or sensitive information before redistribution.

## License and attribution

Code and dataset licensing must be checked separately. This repository does not yet include a license file; public availability alone does not grant unrestricted reuse. Consult the dataset's licensing information and the terms of any constituent assets before use or redistribution.

## Questions

For reproducibility questions, [open an issue](https://github.com/Alibaba-YuFeng/MMA-SafetyBench/issues) with the script name, commit, prompt condition, model identifier, and a sanitized error log. Do not include API keys or private data.
