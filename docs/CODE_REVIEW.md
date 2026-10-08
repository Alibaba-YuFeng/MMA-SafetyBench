# Code review and remaining gaps

Reviewed upstream snapshot: `3a424ee` (2026-10-08). Scope: all 15 Python files,
repository setup, metadata structure, and offline paths. No paid model evaluation
was performed. Dataset records and historical paper results are not rewritten.

Local validation on 2026-10-08:

- 19 offline regression tests passed, including CLI partial/error accounting.
- Ruff checks, formatting checks, Python compilation, and dependency checks passed.
- The local release passed task-count and image-decoding preflight for GUI 209,
  Mobile 283, invoices 100, resumes 100, and Web 200: 892 tasks in total.
- AST comparison of model-facing prompt assignments against the upstream snapshot
  found no changes across the 15 Python scripts.
- No live provider evaluation or end-to-end video validation was performed.
- CI configuration is included, but remote CI has not run for this local branch.

## Implemented repairs

| Area | Upstream issue | Repair |
| --- | --- | --- |
| Configuration | Placeholder credentials and machine paths embedded in scripts | Environment configuration and dataset-root option |
| Imports | Imports created directories, redirected stdout, or constructed API clients | Main adapters import without those side effects |
| API requests | Inconsistent base/full endpoint handling; errors returned as model text | Shared endpoint normalization, bounded retries, explicit failures |
| Invoice score | Score 3 was collapsed to 1 | Strict 1/3/5 parsing |
| Judge failures | Missing/invalid judgments silently became 1 or a guessed fallback | Error state, null scores |
| GUI | Denominator reduced by skipped samples; zero x-coordinate discarded; inconsistent base/defense ASR | Fixed count 209; preserve coordinate zero; both variants use coordinate-target overlap |
| Mobile | Empty negative-button label matched every target string | Only nonempty real labels participate in the override |
| Mobile image | Random bytes appended after PNG end marker | Deterministic valid PNG encoding |
| Assets | Basename/path guesses and silent history-image omission | Domain-scoped resolution, ambiguity failure, all-image preflight |
| Web images | Truncated-file loading and incorrect raw MIME fallback | Strict decode and actual JPEG encoding |
| Results | Shared model output directories and permissive stale-result reuse | Explicit empty run directories and per-run configuration |
| Video | Invalid fallback tool-message sequence; unbounded embedding request | Matching assistant tool-call metadata and request timeout |
| Video judge | API failure could turn into a fabricated score | Explicit error instead of fallback score |
| Maintenance | No installation guide, tests, or declared dependencies | Package metadata, usage guide, regression tests, CI |

## Requires author input / original artifacts

1. **Missing video renderer.** `render_poison_frames.py` is imported upstream but
   absent from the repository. Restore the original implementation; it was not
   invented during cleanup. Video remains unverified end to end.
2. **Historical GUI defense results need reconciliation.** As confirmed by the
   author, both code paths now use coordinates for ASR; judge scores remain for
   SCR. The upstream defense script used the judge's 5. Recompute old results from
   saved coordinates/`coord_hit`, or re-run if unavailable; paper numbers are unchanged.
3. **Video attack scope.** `tools.py` includes a guided textual prompt and an
   administrative tool-response wrapper as well as a visual overlay. This is
   observable in the supplied code and should be reconciled with the paper's
   threat model. Cleanup does not remove that behavior or assert a purely visual
   attack. The frame selection is also stochastic.
4. **Mobile output contract.** The base script expects JSON fields but its original
   system prompt does not explicitly require that JSON schema. Parse failures now
   remain errors. Adding a new formatting prompt would be an experiment change.
5. **License.** There is no upstream code license in this snapshot. Maintainers
   must approve one; a license was not chosen on their behalf.
6. **Reproduction assets.** GitHub contains code and metadata, not the complete
   image/video release. Use the corresponding dataset release and record which
   version was evaluated. Legacy JSON retains original machine paths; the loader
   resolves compatible local assets without rewriting experimental text.

## Security scope

A targeted scan of tracked Python/JSON/JSONL found no non-placeholder `sk-` key
pattern or common signed-URL credential parameter. This is not a comprehensive
secret or privacy certification. Logs and future API responses must be reviewed
before publication. Revoking any previously exposed credentials and purging
historical commits, if needed, are separate actions.
