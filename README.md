# Hierarchical Image Classification (Unified H-CAST + HT-CapsNet)

This repo unifies two hierarchical models behind one PyTorch training/eval pipeline:

- H-CAST (upstream commit: `b1a222bb32da5caf48691b5987d56b7483801907`)
- HT-CapsNet (upstream commit: `8a0ea23f3e6b68b75d8add07674b4b0288380417`)

Single CLI entrypoint:

```bash
python -m train.train --config configs/hcast.yaml
python -m train.train --config configs/capsnet.yaml
```

Dataset-ready configs:

```bash
python -m train.train --config configs/cifar100.yaml
python -m train.train --config configs/cub200.yaml
python -m train.train --config configs/aircraft.yaml
python -m train.train --config configs/inat21mini.yaml
```
## Setup
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
## File Reference

Full file-by-file documentation is available at:

- `docs/FILE_DOCUMENTATION.md`

## Upstream File Mapping

| Upstream | Local |
|---|---|
| `hcast/cast_models/cast_deit_hier.py` | `models/hcast/internal/cast_deit_hier.py` |
| `hcast/cast_models/graph_pool.py` | `models/hcast/internal/graph_pool.py` |
| `hcast/cast_models/modules.py` | `models/hcast/internal/modules.py` |
| `hcast/cast_models/utils.py` | `models/hcast/internal/utils.py` |
| `ht-capsnet/src/model_arch/HTRCapsNet.py` (routing logic) | `models/ht_capsnet/routing.py` + `models/ht_capsnet/model.py` (PyTorch translation) |
| `ht-capsnet/src/models.py` (loss/config ideas) | `models/ht_capsnet/losses.py` + `models/ht_capsnet/factory.py` |

## Unified APIs

### Dataset API
`build_dataloader(cfg, split)` returns:

1. `dataloader`
2. `num_classes_per_level`
3. `taxonomy` (`parent_of` map when available)

Each sample is:

```python
(image_tensor, labels_tensor[L], meta_dict)
```

### Model API
`build_model(cfg, num_classes_per_level, taxonomy)` returns `torch.nn.Module`.

Each model forward returns:

```python
{
  "logits_per_level": [Tensor[B, C0], ..., Tensor[B, CL-1]],
  ...
}
```

### Metrics logged

- `acc_level_0 ... acc_level_L-1`
- `acc_path`
- `inconsistency_rate` (when taxonomy exists)
- `tice_like` (when taxonomy exists)

## Dataset Adapters

Implemented adapters:

- `datasets/breeds.py`
- `datasets/cifar100.py`
- `datasets/cub.py`
- `datasets/aircraft.py`
- `datasets/inat.py`

Supported dataset names:

- `breeds`
- `cifar100`, `cifar-100`
- `cub`, `cub-200-2011`
- `aircraft`, `fgvc-aircraft`
- `inat`, `inat21-mini`, `inat21_mini`

Priority order per dataset:

1. Use `dataset.annotations.{train,val,test}` JSON/TXT if present.
2. Use dataset-specific canonical formats.
3. If enabled, use synthetic fallback (`dataset.allow_synthetic_fallback: true`) for smoke tests.

### Expected canonical formats

- CIFAR-100:
  - Loaded via `torchvision.datasets.CIFAR100`.
  - Canonical hierarchy is 2-level `coarse -> fine` (20/100).

- CUB-200-2011:
  - Pre-split folders: `root/train/<class>/...` and `root/test/<class>/...`.
  - Also supported: `root/images_split/{train,test}/...`.
  - Official raw layout fallback: `images.txt`, `image_class_labels.txt`, `train_test_split.txt`, and `images/` under `root` or `root/CUB_200_2011`.

- FGVC-Aircraft:
  - Official txt files under `.../data/`:
    - `images_variant_train.txt`
    - `images_variant_val.txt`
    - `images_variant_test.txt`
  - Fallback: `images_variant_trainval.txt` + deterministic train/val split.

- iNat21-Mini:
  - Primary list format: `path species family order`.
  - Expected hierarchy: 3-level `order -> family -> species`.
  - Auto-detects common split-list names in `dataset.root`, `dataset.root/data`, and `./data`.

### Split fallback policy

If a dataset has no explicit validation split:

- A deterministic stratified validation subset is created from train data.
- Controls:
  - `dataset.val_split_ratio` (default: `0.1`)
  - `dataset.split_seed` (default fallback: `train.seed`)

For iNat test split fallback:

- Use explicit test file if present.
- Else use val file.
- Else use the deterministic val subset derived from train/trainval.

## Taxonomy JSON Schema

Expected minimal structure:

```json
{
  "levels": ["level0", "level1", "level2"],
  "parent_of": {
    "1": {"child_id": 0},
    "2": {"child_id": 4}
  }
}
```

`parent_of[level][child] = parent` where `level` is child level index.

## Reproducibility

- Seed: `train.seed`
- Determinism: `runtime.deterministic`
- AMP: `train.amp`
- Checkpoints: `latest.pt`, `best.pt`
- Resume: `train.resume`

## Notes

- H-CAST path uses upstream internals directly under `models/hcast/internal/`.
- HT-CapsNet upstream is TensorFlow; this repo provides a PyTorch translation of the taxonomy-guided routing path while preserving the unified interfaces and training flow.
- If optional deps for full H-CAST stack are missing (e.g., `timm`/`dgl`), model code falls back to a lightweight path so the CLI remains runnable.
