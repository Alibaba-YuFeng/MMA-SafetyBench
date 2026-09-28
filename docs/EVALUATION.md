# Evaluation guide

[Back to README](../README.md)

This guide describes the current repository implementation. It does not certify that every runner reproduces a paper table unchanged.

## Runner map

All paths below are relative to `Benchmark_Dataset/`.

| Domain | Runner | Alternative runner |
| :--- | :--- | :--- |
| GUI | `01_Web_GUI_Testing/gui_attack_pipeline/gui_attack_pipeline/scripts/run_frontend_benchmark.py` | `run_frontend_benchmark_prompt.py` in the same directory |
| Document | `02_Document_Analysis/run_doc_eval.py` | `run_doc_eval_defense.py` |
| Resume | `02_Document_Analysis/run_resume_eval.py` | `run_resume_eval_defense.py` |
| Web | `03_web/run_benchmark_eval.py` | `run_benchmark_eval_defense.py` |
| Mobile | `04_Mobile_Navigation/run_mobile.py` | `run_mobile_defense.py` |
| Video | `05_Video_Agent/dvd_agent/run_final_video_eval_v3.py` | — |

Inspect the actual system prompt in each file. Filenames alone do not establish whether a run is neutral, task-priority, or defended. In particular, the GUI `_prompt.py` runner is not a neutral-prompt control.

## Before running

1. Obtain the relevant dataset assets from the [dataset repository](https://huggingface.co/datasets/Alibaba-YuFeng/MMA-SafetyBench).
2. Open the chosen script and configure its victim and judge model names, API URLs, and credentials. Check how the client constructs its URL: a base URL and a complete endpoint are not interchangeable.
3. Set all input and output paths for your machine. Several scripts retain home-directory or original-machine paths.
4. Confirm that metadata paths resolve to real assets. Do not silently exclude missing files from a reported denominator.
5. Use a new output directory for each model and prompt condition. Some runners reuse existing result files.
6. Run a small, explicitly selected subset before a full paid API evaluation. Keep the subset separate from the benchmark's full evaluation set.

After configuration, a web evaluation can be launched from the repository root with:

```bash
python Benchmark_Dataset/03_web/run_benchmark_eval.py
```

This command requires valid API configuration and the referenced images; it is not an offline smoke test.

## Required assets and path conventions

| Runner family | Inputs referenced by the source |
| :--- | :--- |
| GUI | Under `GUI_ROOT`: `04_attack_jsons/` and `03_poisoned_frontend/` |
| Document | `GEPA_WEB_STYLE_DIVERSE.json` and its referenced images |
| Resume | `GEPA_FULL_DIMENSION_ATTACK_V3.json`, `natural_boss_attacks/`, and `base_resumes/` |
| Web | `dataset_balanced_200.json` and its referenced images |
| Mobile | Under `MOBILE_ROOT`: `2_unified_meta_jsons/`, `adversarial_privilege_results/`, and `1_unified_raw_images/` |
| Video | `lvbench_video/`, `data/video_info.meta.jsonl`, and the local rendering module imported by `tools.py` |

These paths describe the code's expectations, not a verified Hugging Face archive layout. The video metadata file is currently stored beside the video scripts, while the runner expects it under `data/`. Also, `tools.py` imports `render_poison_frames`, which is absent from this GitHub checkout. Resolve both before running the video pipeline; do not substitute an invented rendering implementation.

## Implementation notes

### GUI success criteria

The base GUI runner falls back to coordinate-hit success when the judge response lacks `is_attack_success`. The alternative `_prompt.py` runner sets success from judge score 5. These outputs are not automatically comparable as a single ASR definition. A predicted coordinate hit does not demonstrate an executed click.

### Denominators and failures

The GUI scripts print rates over `valid_responses`. Missing assets and unsuccessful requests can therefore make a run's denominator differ from the paper's nominal sample count. Record requested samples, completed samples, missing assets, API failures, and scoring failures separately. Do not equate a completed-response rate with a fixed-denominator paper result without reconciliation.

### Semantic scores

The base GUI runner does not persist the numeric semantic score in its per-sample result record. Those records alone are insufficient to recompute SCR. Preserve judge scores and the corresponding raw responses in any validated release used for SCR reporting.

## Reproducibility record

For every reported run, retain:

- Repository commit and dataset version or manifest.
- Exact model identifiers, API provider, inference parameters, and run date.
- Complete system/user prompts and expected output format.
- Judge prompt, judge model, score interpretation, and success rule.
- Sample IDs, raw outputs, parsed actions, scores, and error status.
- Numerator, denominator, exclusion policy, and aggregation script.
- Dependency versions and relevant hardware/runtime details.

No model inference was run as part of this documentation update, and this dependency list is not a validated reproduction environment.
