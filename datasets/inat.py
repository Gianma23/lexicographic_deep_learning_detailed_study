import json
import tarfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseHierDataset, split_train_val_samples


class INat19Dataset(BaseHierDataset):
    """iNaturalist 2019 adapter (Visipedia inat_comp/2019 format)."""

    def default_levels(self) -> List[str]:
        """Default hierarchy names used when config does not provide levels."""
        return ["family", "genus", "species"]

    def load_samples(self) -> List[Dict[str, Any]]:
        """Load iNat19 using explicit manifests or official annotation fallbacks."""
        split_policy = self.cfg.dataset.get("split_policy", "official_val_test")
        if split_policy == "explicit":
            ann = self._configured_annotation_file(self.split)
            if ann is None:
                return []
            return self._read_inat19_annotations(ann, split_hint=self.split)

        train_ann = self._find_preferred_annotation_file("train")
        val_ann = self._find_preferred_annotation_file("val")

        if self.split in {"train", "val"}:
            if train_ann is None:
                return []
            pool = self._read_inat19_annotations(train_ann, split_hint="train")
            return split_train_val_samples(pool, split=self.split, cfg=self.cfg, stratify_level=-1)

        # test split: use official validation labels as the held-out test set.
        if val_ann is not None:
            test_samples = self._read_inat19_annotations(val_ann, split_hint="val")
            if test_samples:
                return test_samples
            warnings.warn(
                "Validation annotation file exists but produced no labeled samples; trying configured test fallback.",
                RuntimeWarning,
            )

        test_ann = self._find_preferred_annotation_file("test")
        if test_ann is not None:
            test_samples = self._read_inat19_annotations(test_ann, split_hint="test")
            if test_samples:
                return test_samples

        if train_ann is not None:
            pool = self._read_inat19_annotations(train_ann, split_hint="train")
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
        """Resolve official iNat 2019 annotation files for the requested split."""
        names_by_split = {
            "train": [
                "train2019.json",
                "train2019.json.tar.gz",
            ],
            "val": [
                "val2019.json",
                "val2019.json.tar.gz",
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

    def _read_inat19_annotations(self, ann_path: Path, split_hint: str) -> List[Dict[str, Any]]:
        """Read official iNat19 annotations from JSON or JSON-in-tar files."""
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
            meta = row.get("meta")
            if isinstance(meta, dict):
                meta = dict(meta)
            else:
                meta = dict(row)
            out.append({"image": image_path, "labels": labels, "meta": meta})
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
        """Parse Visipedia iNat19 COCO-style annotations into dataset samples."""
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
        family_keys = set()
        genus_keys = set()
        for category in categories:
            if not isinstance(category, dict):
                continue
            try:
                category_id = int(category.get("id"))
            except (TypeError, ValueError):
                continue

            family_key = self._normalize_taxon_name(category.get("family"))
            genus_key = self._normalize_taxon_name(category.get("genus"))
            if family_key and genus_key:
                family_keys.add(family_key)
                genus_keys.add((family_key, genus_key))
            category_by_id[category_id] = category

        family_to_id = {name: idx for idx, name in enumerate(sorted(family_keys))}
        genus_to_id = {name: idx for idx, name in enumerate(sorted(genus_keys))}

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

            family_key = self._normalize_taxon_name(category.get("family"))
            genus_key = self._normalize_taxon_name(category.get("genus"))
            if not family_key or not genus_key:
                continue
            family = family_to_id.get(family_key)
            genus = genus_to_id.get((family_key, genus_key))
            if family is None or genus is None:
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
                    "labels": [family, genus, species],
                    "meta": {
                        "source": "inat19_coco",
                        "annotation_file": str(ann_path),
                        "annotation_id": ann.get("id"),
                        "image_id": image_id,
                        "category_id": species,
                        "family_name": category.get("family"),
                        "genus_name": category.get("genus"),
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
        if ann_lower.startswith("train2019") or ann_lower.startswith("val2019"):
            return "train_val2019"
        if ann_lower.startswith("train") or ann_lower.startswith("val"):
            return "train_val2019"

        by_split = {
            "train": "train_val2019",
            "val": "train_val2019",
            "test": "train_val2019",
        }
        return by_split.get(split_hint, "")

    def _resolve_image_path(
        self,
        rel_path: str,
        ann_path: Path,
        split_hint: str,
        category_dir: str = "",
    ) -> Path:
        """Resolve iNat19 image paths across official train_val2019 layouts."""
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
