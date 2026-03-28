import json
import tarfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseHierDataset, split_train_val_samples


class INatDataset(BaseHierDataset):
    """iNaturalist 2021 mini adapter (Visipedia inat_comp/2021 format)."""

    def default_levels(self) -> List[str]:
        """Default hierarchy names used when config does not provide levels."""
        return ["order", "family", "species"]

    def load_samples(self) -> List[Dict[str, Any]]:
        """Load iNat21-mini using split policy: train->(train,val), official val->test."""
        train_ann = self._find_preferred_annotation_file("train")
        val_ann = self._find_preferred_annotation_file("val")

        if self.split in {"train", "val"}:
            if train_ann is None:
                return []
            pool = self._read_inat21_annotations(train_ann, split_hint="train")
            return split_train_val_samples(pool, split=self.split, cfg=self.cfg, stratify_level=-1)

        # test split: use official validation labels as the held-out test set.
        if val_ann is not None:
            test_samples = self._read_inat21_annotations(val_ann, split_hint="val")
            if test_samples:
                return test_samples
            warnings.warn(
                "Validation annotation file exists but produced no labeled samples; trying test/public_test fallback.",
                RuntimeWarning,
            )

        test_ann = self._find_preferred_annotation_file("test")
        if test_ann is not None:
            test_samples = self._read_inat21_annotations(test_ann, split_hint="test")
            if test_samples:
                return test_samples

        if train_ann is not None:
            pool = self._read_inat21_annotations(train_ann, split_hint="train")
            return split_train_val_samples(pool, split="val", cfg=self.cfg, stratify_level=-1)
        return []

    def _find_preferred_annotation_file(self, split_name: str) -> Optional[Path]:
        """Prefer config-provided annotation path, then fallback to official file names."""
        configured = self._configured_annotation_file(split_name)
        if configured is not None:
            return configured
        return self._find_official_annotation_file(split_name)

    def _configured_annotation_file(self, split_name: str) -> Optional[Path]:
        """Resolve a configured annotation file for an arbitrary split key."""
        ann_cfg = self.cfg.dataset.get("annotations", {})
        split_file = ann_cfg.get(split_name)
        if not split_file:
            return None
        path = self.root / str(split_file)
        if path.exists():
            return path
        return None

    def _find_official_annotation_file(self, split_name: str) -> Optional[Path]:
        """Resolve official iNat 2021 annotation files for the requested split."""
        names_by_split = {
            "train": [
                "train_mini.json",
                "train_mini.json.tar.gz",
                "train.json",
                "train.json.tar.gz",
            ],
            "val": [
                "val.json",
                "val.json.tar.gz",
            ],
            "test": [
                "public_test.json",
                "public_test.json.tar.gz",
                "test.json",
                "test.json.tar.gz",
            ],
        }

        roots = [self.root, self.root / "data", Path("data")]
        candidates: List[Path] = []
        for root in roots:
            for name in names_by_split.get(split_name, []):
                candidates.append(root / name)
        return self._find_existing(candidates)

    @staticmethod
    def _find_existing(candidates: List[Path]) -> Optional[Path]:
        """Return the first existing path from a list of candidates."""
        for path in candidates:
            if path.exists():
                return path
        return None

    def _read_inat21_annotations(self, ann_path: Path, split_hint: str) -> List[Dict[str, Any]]:
        """Read official iNat21-mini annotations from JSON or JSON-in-tar files."""
        payload = self._load_json_payload(ann_path)
        if not payload:
            return []

        if isinstance(payload, dict) and "images" in payload and "categories" in payload:
            return self._read_coco_style_annotations(payload, ann_path, split_hint)

        # Fallback to the repository's normalized annotation format.
        rows = payload.get("samples", []) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            return []

        out: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            labels = row.get("labels") or row.get("levels")
            image_rel = row.get("image") or row.get("path")
            if labels is None or image_rel is None:
                continue
            labels = [int(x) for x in labels][: self.depth]
            if len(labels) != self.depth:
                continue
            image_path = self._resolve_image_path(str(image_rel), ann_path=ann_path, split_hint=split_hint)
            out.append({"image": image_path, "labels": labels, "meta": dict(row)})
        return out

    @staticmethod
    def _normalize_taxon_name(value: Any) -> str:
        """Normalize taxonomy strings for stable deterministic ID mapping."""
        return str(value or "").strip().lower()

    def _read_coco_style_annotations(
        self,
        payload: Dict[str, Any],
        ann_path: Path,
        split_hint: str,
    ) -> List[Dict[str, Any]]:
        """Parse Visipedia iNat21 COCO-style annotations into dataset samples."""
        images = payload.get("images", [])
        categories = payload.get("categories", [])
        annotations = payload.get("annotations", [])

        if not isinstance(images, list) or not isinstance(categories, list) or not isinstance(annotations, list):
            return []
        if not annotations:
            # public_test.json has no labels by design.
            return []

        image_by_id: Dict[int, Dict[str, Any]] = {}
        for image in images:
            if not isinstance(image, dict):
                continue
            try:
                image_id = int(image.get("id"))
            except (TypeError, ValueError):
                continue
            image_by_id[image_id] = image

        category_by_id: Dict[int, Dict[str, Any]] = {}
        order_keys = set()
        family_keys = set()
        for category in categories:
            if not isinstance(category, dict):
                continue
            try:
                category_id = int(category.get("id"))
            except (TypeError, ValueError):
                continue

            order_key = self._normalize_taxon_name(category.get("order"))
            family_key = self._normalize_taxon_name(category.get("family"))
            if order_key and family_key:
                order_keys.add(order_key)
                family_keys.add((order_key, family_key))
            category_by_id[category_id] = category

        order_to_id = {name: idx for idx, name in enumerate(sorted(order_keys))}
        family_to_id = {name: idx for idx, name in enumerate(sorted(family_keys))}

        samples: List[Dict[str, Any]] = []
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            try:
                image_id = int(ann.get("image_id"))
                species = int(ann.get("category_id"))
            except (TypeError, ValueError):
                continue

            image = image_by_id.get(image_id)
            category = category_by_id.get(species)
            if image is None or category is None:
                continue

            order_key = self._normalize_taxon_name(category.get("order"))
            family_key = self._normalize_taxon_name(category.get("family"))
            if not order_key or not family_key:
                continue
            order = order_to_id.get(order_key)
            family = family_to_id.get((order_key, family_key))
            if order is None or family is None:
                continue

            file_name = str(image.get("file_name", "")).strip()
            if not file_name:
                continue
            category_dir = str(category.get("image_dir_name", "")).strip()
            image_path = self._resolve_image_path(
                file_name,
                ann_path=ann_path,
                split_hint=split_hint,
                category_dir=category_dir,
            )

            samples.append(
                {
                    "image": image_path,
                    "labels": [order, family, species],
                    "meta": {
                        "source": "inat21_coco",
                        "annotation_file": str(ann_path),
                        "annotation_id": ann.get("id"),
                        "image_id": image_id,
                        "category_id": species,
                        "order_name": category.get("order"),
                        "family_name": category.get("family"),
                        "species_name": category.get("name"),
                        "file_name": file_name,
                    },
                }
            )

        return samples

    @staticmethod
    def _load_json_payload(path: Path) -> Any:
        """Load JSON from plain .json files or from .json inside a .tar(.gz) archive."""
        lower_name = path.name.lower()
        if lower_name.endswith(".json"):
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)

        if lower_name.endswith(".tar.gz") or lower_name.endswith(".tgz") or lower_name.endswith(".tar"):
            with tarfile.open(path, "r:*") as tar:
                members = [m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith(".json")]
                if not members:
                    return {}

                target = members[0]
                wanted_name = lower_name.replace(".tar.gz", "").replace(".tgz", ".json").replace(".tar", "")
                for member in members:
                    member_name = Path(member.name).name.lower()
                    if member_name == wanted_name:
                        target = member
                        break

                extracted = tar.extractfile(target)
                if extracted is None:
                    return {}
                return json.loads(extracted.read().decode("utf-8"))

        return {}

    @staticmethod
    def _split_dir_for_context(split_hint: str, ann_name: str) -> str:
        """Infer expected image subdirectory from split hint and annotation file name."""
        ann_lower = ann_name.lower()
        if "train_mini" in ann_lower:
            return "train_mini"
        if ann_lower.startswith("train"):
            return "train"
        if ann_lower.startswith("val"):
            return "val"
        if "public_test" in ann_lower:
            return "public_test"
        if ann_lower.startswith("test"):
            return "public_test"

        by_split = {
            "train": "train_mini",
            "val": "val",
            "test": "public_test",
        }
        return by_split.get(split_hint, "")

    def _resolve_image_path(
        self,
        rel_path: str,
        ann_path: Path,
        split_hint: str,
        category_dir: str = "",
    ) -> Path:
        """Resolve iNat image paths across official train_mini/val/public_test layouts."""
        rel = Path(rel_path)
        if rel.is_absolute():
            return rel

        split_dir = self._split_dir_for_context(split_hint, ann_path.name)
        bases = [self.root, ann_path.parent, self.root / "data", Path("data")]

        candidates: List[Path] = []
        for base in bases:
            candidates.append(base / rel)
            if split_dir:
                candidates.append(base / split_dir / rel)

            if category_dir and rel.name:
                cat_dir = Path(category_dir)
                candidates.append(base / cat_dir / rel.name)
                if split_dir:
                    candidates.append(base / split_dir / cat_dir / rel.name)

        seen = set()
        deduped_candidates: List[Path] = []
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            deduped_candidates.append(candidate)

        for candidate in deduped_candidates:
            if candidate.exists():
                return candidate

        return deduped_candidates[0] if deduped_candidates else self.root / rel
