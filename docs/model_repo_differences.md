# Model-by-Model Upstream Delta Log for Thesis Reproduction

Date: May 19, 2026

## Purpose

This document summarizes how the implementations in this repository differ from the original model sources used for reproduction work.  
The goal is to provide thesis-ready traceability for what was changed, what was preserved, and why.

The comparison includes both:

- `Structural` changes: required by the unified PyTorch training/evaluation framework.
- `Intentional` changes: deliberate research or reproducibility choices introduced in this repository.

## Upstream Baselines (Pinned Snapshot)

Snapshot policy: upstream `HEAD` references were pinned on **May 19, 2026**.

| Model | Upstream baseline | Pinned reference (2026-05-19) | Comparison mode |
|---|---|---|---|
| H-CAST | `https://github.com/pseulki/HCAST` | `b1a222bb32da5caf48691b5987d56b7483801907` | repo-to-repo |
| HT-CapsNet | `https://github.com/tasrif-khondaker/HT-CapsNet` | `8a0ea23f3e6b68b75d8add07674b4b0288380417` | repo-to-repo |
| HRN | `https://github.com/MonsterZhZh/HRN` | `59944e48fcbf41cc475402c8b9cb6af301006399` | repo-to-repo |
| Hier-COS | `https://github.com/Depanshu-Sani/Hier-COS` | `122b01dff393d3b562ad2daac494496fda65131c` | repo-to-repo |
| LH-DNN | arXiv `2409.16956` (no official code repo) | N/A | paper-to-code |

Pinning command used: `git ls-remote <repo_url> HEAD`.

## Methodology

- Differences are labeled as `Structural` or `Intentional`.
- `Structural` deltas are mostly due to unification into one codebase (`python -m train.train`) with shared data loading, optimization, logging, checkpointing, and evaluation.
- `Intentional` deltas are deliberate design choices to support clean comparisons, parity presets, diagnostics, and lexicographic experiments.
- For LH-DNN, comparison is against the paper specification because no official upstream repository exists.

## Changes Affecting All Models

- `Structural`: Unified training entrypoint and lifecycle (`train/val/test`) for all families.
  - Why: keep one reproducible execution path and remove model-specific training scripts.
- `Structural`: Validation-based checkpoint selection is unified and mode-specific (`topdown` and `independent` are ranked separately).
  - Ranking order: `FPA` (higher), `TICE` (lower), `weighted AP` (higher).
  - Why: avoid selecting one model with a different criterion than another.
- `Structural`: Final test reporting is always done from two best checkpoints (top-down-selected and independent-selected).
  - Why: preserve decoder-dependent fairness and avoid mixing selection modes.
- `Structural`: Shared runtime controls are applied to every model (determinism, resume metadata, common output artifacts).
  - Common artifacts: `config_resolved.yaml`, `run_log.jsonl`, `best_topdown.pt`, `best_independent.pt`, `test_metrics.yaml`.
- `Structural`: AMP is exposed as a unified runtime switch (`train.amp`, CUDA-only gate).
  - Why: consistent mixed-precision behavior across models without model-specific AMP codepaths.
- `Structural`: Shared dataset/transform abstraction and taxonomy handling (including deterministic split logic and taxonomy inference fallbacks).
  - Why: make dataset protocol consistent across baselines while preserving model-specific constraints.
- `Intentional`: Common diagnostic infrastructure (gradient/parameter logs, lexicographic hooks) was integrated where compatible.
  - Why: support thesis analysis beyond raw accuracy tables.

## Model-Specific Deltas

### H-CAST (`hcast`)

- `Structural`: Upstream CAST internals are vendored under `models/hcast/internal/` and wrapped by a local `HCASTModel` adapter through `timm.create_model`.
  - Why: stabilize upstream dependencies and fit the unified model interface.
- `Structural`: Class order adaptation is explicit (repo uses coarse-to-fine ordering; upstream H-CAST internals expect fine-to-coarse indexing).
  - Why: keep repository-wide label conventions consistent.
- `Intentional`: Hierarchical Constraint Cascade (HCC) was added as an optional output-space affine projector (`models/hcast/hard_hierarchy.py`) with schedule controls (`step`, `linear`, `exp`, `tanh`) and diagnostics.
  - Why: enable hard-hierarchy intervention studies and step@0/step@80 analyses.
- `Intentional`: HCC diagnostics (`proj_constraint_alpha`, residual before/after, etc.) are logged for activation-time verification.
  - Why: avoid inferring HCC state from run names only.
- `Structural`: H-CAST was integrated into shared checkpointing, dual-mode evaluation, and unified metrics stack.
  - Why: preserve comparability with non-H-CAST baselines.

### LH-DNN (`lhdnn`)

- `Structural`: Implemented as paper-derived PyTorch code (no official upstream repo), with fixed large topology and always-on projection/advantage pathways.
  - Why: reproduce the core LH-DNN method from published specification.
- `Structural`: Taxonomy is mandatory in the local implementation (advantage topology requires explicit parent mappings).
  - Why: enforce consistency with hierarchical constraints assumed by the method.
- `Structural`: Integrated into the unified trainer and evaluator (shared splits, checkpoint logic, decoder metrics, logging).
  - Why: enable apples-to-apples comparison against other models.
- `Intentional`: Dataset presets beyond paper-reported CIFAR are explicit extrapolations (CUB/Aircraft), marked in configs.
  - Why: extend comparative experiments while keeping extrapolation status transparent.
- `Structural`: Loss path accepts both hard labels and shared soft-target container format from the common pipeline.
  - Why: interface compatibility with the global training stack.

### HT-CapsNet (`ht_capsnet`)

- `Structural`: Original TensorFlow-oriented implementation is ported to a PyTorch model (`models/ht_capsnet/model.py`) while preserving capsule-routing design intent.
  - Why: run inside the repository-wide PyTorch framework.
- `Structural`: Backbone flexibility (`custom` conv stack or `efficientnet_b7`) is implemented with compatibility fallbacks.
  - Why: maintain parity-focused defaults while supporting practical training environments.
- `Intentional`: Parity preprocessing is encoded in shared config/transforms (`fixed_resize_only` + per-image `standardscaler` normalization).
  - Why: keep preprocessing behavior close to upstream script expectations.
- `Intentional`: Loss weighting modes (`dynamic`, `static`, `none`) are explicit and configurable in local code.
  - Why: make hierarchy weighting behavior inspectable and controllable in experiments.
- `Structural`: Reproducibility guards are enforced at build time (`train.seed` required, `runtime.deterministic: true` required).
  - Why: prevent non-reproducible runs from silently entering benchmark comparisons.
- `Structural`: AMP is available via shared runtime (`train.amp: true` in shipped HT-CapsNet configs).
  - Why: align execution controls with the rest of the repository.

### HRN (`hrn`)

- `Structural`: Local model/loss follows upstream HRN full-label branch semantics, but is served through the shared training stack.
  - Why: preserve architectural parity while unifying orchestration.
- `Intentional`: Protocol divergence is explicit: repo uses train/val/test with validation-based checkpoint selection instead of direct train/test-only workflow.
  - Why: cleaner evaluation discipline across all baselines.
- `Intentional`: Scope is limited to full-label mode; partial-label branch behavior is intentionally out of scope in this pass.
  - Why: avoid mixing protocol regimes inside one comparison suite.
- `Intentional`: CIFAR-100 HRN preset is a local extrapolation (upstream HRN does not provide CIFAR-100 experiments).
  - Why: include HRN in the same dataset grid used for other models.
- `Structural`: Mixup/cutmix soft-target training paths are rejected for parity loss behavior.
  - Why: keep loss semantics aligned with HRN parity assumptions.

### Hier-COS (`hiercos`)

- `Structural`: Local code keeps upstream-style fixed-frame/taxonomy-subspace formulation and KL+regularization objective path (`model.loss: kl_reg`) as the parity baseline.
  - Why: preserve original method core before ablations.
- `Intentional`: Added `model.loss: per_level_ce` ablation and `model.ce_weight_mode` (`equal`, `kl_leaf`, `kl_coarse`).
  - Why: expose three differentiable level losses for gradient and lexicographic diagnostics.
- `Intentional`: CIFAR protocol keeps repository hierarchy depth (`3`) rather than upstream 5-level CIFAR protocol.
  - Why: maintain cross-model hierarchy consistency inside this repository.
- `Intentional`: CUB-200 Hier-COS preset is explicitly marked as local extrapolation.
  - Why: upstream Hier-COS does not report CUB experiments.
- `Intentional`: Aircraft preprocessing includes explicit bottom-banner crop support (`crop_bottom_pixels: 20`) for parity.
  - Why: replicate known preprocessing behavior from upstream scripts.
- `Structural`: AMP is enabled in shipped Hier-COS configs (`train.amp: true`) via shared runtime controls.
  - Why: standardized mixed-precision support in the unified pipeline.

## Thesis Synthesis

### Evidence-Backed Facts

- All five model families are executed under one unified PyTorch pipeline with shared checkpointing and evaluation rules.
- Top-down and independent decoding are both first-class evaluation modes, with separate best-checkpoint selection and final test reporting.
- H-CAST includes local HCC extensions not present in the original H-CAST upstream repository.
- HT-CapsNet is implemented as a PyTorch port of a TensorFlow-origin baseline.
- Hier-COS includes a local CE ablation path (`per_level_ce`) beyond the paper-aligned KL baseline.
- LH-DNN comparison is necessarily paper-to-code (no official upstream code repository).

### Interpretation for Thesis Writing

- Most large deltas are not arbitrary model edits; they are consequences of enforcing one evaluation protocol for all baselines.
- Intentional local additions (HCC controls, Hier-COS CE ablation, diagnostics) were introduced to answer research questions about hierarchy constraints and lexicographic behavior, not to claim paper-faithful parity.
- Reported reproduction outcomes should be interpreted as results under this repository protocol, with parity to upstream logic where feasible and explicit local extrapolations where upstream coverage is missing.

### Limitations and Threats to Validity

- Upstream drift: pinned SHAs are fixed on May 19, 2026; upstream repositories may change afterward.
- LH-DNN source limitation: absence of official code forces paper-to-code reconstruction risk.
- Extrapolation risk: HRN CIFAR and Hier-COS CUB settings are local extrapolations, not upstream-reported benchmarks.
- Protocol dependence: validation-based dual-checkpoint selection differs from some original repos and can alter final test outcomes.
- Decoder dependence: top-down and independent conclusions may differ; they must not be merged into a single claim without mode qualification.

## Local Evidence Anchors

- Shared protocol/runtime: `train/train.py`, `train/engine.py`, `train/runtime/selection.py`, `train/training_logger.py`.
- Dataset/transform abstraction: `datasets/base.py`, `datasets/__init__.py`.
- H-CAST: `models/hcast/model.py`, `models/hcast/hard_hierarchy.py`, `models/hcast/internal/`.
- LH-DNN: `models/lhdnn/model.py`, `models/lhdnn/losses.py`, `configs/lhdnn/*.yaml`.
- HT-CapsNet: `models/ht_capsnet/model.py`, `models/ht_capsnet/losses.py`, `models/ht_capsnet/factory.py`, `configs/capsnet/*.yaml`.
- HRN + Hier-COS parity audit: `docs/hrn_hiercos_alignment.md`.
- Hier-COS local ablations: `models/hiercos/losses.py`, `configs/hiercos/*.yaml`.
