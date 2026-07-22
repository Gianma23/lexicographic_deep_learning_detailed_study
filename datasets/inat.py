import hashlib
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

    def load_taxonomy(self) -> Optional[Dict[str, Any]]:
        """Return category-derived raw taxonomy for official COCO annotations."""
        return getattr(self, "_annotation_taxonomy", None)

    def load_samples(self) -> List[Dict[str, Any]]:
        """Load iNat19 using explicit manifests or official annotation fallbacks."""
        split_policy = self.cfg.dataset.get("split_policy", "official_val_test")
        if split_policy == "explicit":
            ann = self._configured_annotation_file(self.split)
            if ann is None:
                raise ValueError(
                    f"dataset.split_policy=explicit requires dataset.annotations.{self.split}."
                )
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
        path = Path(str(split_file)).expanduser()
        if not path.is_absolute():
            path = self.root / path
        if not path.is_file():
            raise FileNotFoundError(
                f"Configured iNat19 annotation for split={split_name} does not exist: {path}"
            )
        return path

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
            raise ValueError(f"iNat19 annotation payload is empty: {ann_path}")

        if isinstance(payload, dict) and "images" in payload and "categories" in payload:
            return self._read_coco_style_annotations(payload, ann_path, split_hint)

        # Fallback to the repository's normalized annotation format.
        rows = payload.get("samples", []) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise TypeError(f"iNat19 annotation {ann_path} must contain a list of samples.")

        out: List[Dict[str, Any]] = []
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise TypeError(
                    f"Malformed iNat19 annotation {ann_path} row {row_index}: expected a mapping."
                )
            labels = row.get("labels") or row.get("levels")
            image_rel = row.get("image") or row.get("path")
            if labels is None or image_rel is None:
                raise ValueError(
                    f"Malformed iNat19 annotation {ann_path} row {row_index}: "
                    "both labels/levels and image/path are required."
                )
            if not isinstance(labels, (list, tuple)):
                raise TypeError(
                    f"Malformed iNat19 annotation {ann_path} row {row_index}: labels must be a list."
                )
            if len(labels) != self.depth:
                raise ValueError(
                    f"Malformed iNat19 annotation {ann_path} row {row_index}: "
                    f"expected {self.depth} labels, got {len(labels)}."
                )
            try:
                normalized_labels = [int(value) for value in labels]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Malformed iNat19 annotation {ann_path} row {row_index}: labels must be integers."
                ) from exc
            image_path = self._resolve_image_path(str(image_rel), ann_path=ann_path, split_hint=split_hint)
            meta = row.get("meta")
            if isinstance(meta, dict):
                meta = dict(meta)
            else:
                meta = dict(row)
            meta["annotation_file"] = str(ann_path)
            meta["annotation_row"] = row_index
            out.append({"image": image_path, "labels": normalized_labels, "meta": meta})
        return out

    @staticmethod
    def _normalize_taxon_name(value: Any) -> str:
        """Normalize taxonomy strings for stable deterministic ID mapping."""
        return str(value or "").strip().lower()

    @staticmethod
    def _stable_taxon_id(rank: str, value: str) -> int:
        """Encode a taxon name as a split-independent non-negative raw ID."""
        digest = hashlib.sha256(f"inat19:{rank}:{value}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False) & ((1 << 63) - 1)

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
            raise TypeError(
                f"Malformed iNat19 COCO annotation {ann_path}: "
                "`images`, `categories`, and `annotations` must be lists."
            )
        if not annotations:
            # public_test.json has no labels by design.
            return []

        image_by_id: Dict[int, Dict[str, Any]] = {}
        for row_index, image in enumerate(images):
            if not isinstance(image, dict):
                raise TypeError(
                    f"Malformed iNat19 COCO annotation {ann_path} image row {row_index}: "
                    "expected a mapping."
                )
            try:
                image_id = int(image.get("id"))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Malformed iNat19 COCO annotation {ann_path} image row {row_index}: "
                    "missing or non-integer id."
                ) from exc
            if image_id in image_by_id:
                raise ValueError(
                    f"Malformed iNat19 COCO annotation {ann_path} image row {row_index}: "
                    f"duplicate image id {image_id}."
                )
            if image_id < 0:
                raise ValueError(
                    f"Malformed iNat19 COCO annotation {ann_path} image row {row_index}: "
                    f"negative image id {image_id}."
                )
            image_by_id[image_id] = image

        category_by_id: Dict[int, Dict[str, Any]] = {}
        family_keys = set()
        genus_keys = set()
        for row_index, category in enumerate(categories):
            if not isinstance(category, dict):
                raise TypeError(
                    f"Malformed iNat19 COCO annotation {ann_path} category row {row_index}: "
                    "expected a mapping."
                )
            try:
                category_id = int(category.get("id"))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Malformed iNat19 COCO annotation {ann_path} category row {row_index}: "
                    "missing or non-integer id."
                ) from exc
            if category_id in category_by_id:
                raise ValueError(
                    f"Malformed iNat19 COCO annotation {ann_path} category row {row_index}: "
                    f"duplicate category id {category_id}."
                )
            if category_id < 0:
                raise ValueError(
                    f"Malformed iNat19 COCO annotation {ann_path} category row {row_index}: "
                    f"negative category id {category_id}."
                )

            family_key = self._normalize_taxon_name(category.get("family"))
            genus_key = self._normalize_taxon_name(category.get("genus"))
            if not family_key or not genus_key:
                raise ValueError(
                    f"Malformed iNat19 COCO annotation {ann_path} category row {row_index}: "
                    "family and genus are required."
                )
            family_keys.add(family_key)
            genus_keys.add((family_key, genus_key))
            category_by_id[category_id] = category

        family_to_id = {
            name: self._stable_taxon_id("family", name)
            for name in sorted(family_keys)
        }
        genus_to_id = {
            name: self._stable_taxon_id("genus", f"{name[0]}\0{name[1]}")
            for name in sorted(genus_keys)
        }
        if len(set(family_to_id.values())) != len(family_to_id):
            raise RuntimeError("Stable iNat19 family IDs collided; refusing an ambiguous taxonomy.")
        if len(set(genus_to_id.values())) != len(genus_to_id):
            raise RuntimeError("Stable iNat19 genus IDs collided; refusing an ambiguous taxonomy.")

        genus_to_family: Dict[int, int] = {}
        species_to_genus: Dict[int, int] = {}
        for species, category in category_by_id.items():
            family_key = self._normalize_taxon_name(category.get("family"))
            genus_key = self._normalize_taxon_name(category.get("genus"))
            family = family_to_id[family_key]
            genus = genus_to_id[(family_key, genus_key)]
            previous_family = genus_to_family.get(genus)
            if previous_family is not None and previous_family != family:
                raise ValueError(
                    f"Malformed iNat19 COCO annotation {ann_path}: genus {genus_key!r} "
                    "maps to multiple families."
                )
            genus_to_family[genus] = family
            species_to_genus[species] = genus
        self._annotation_taxonomy = {
            "levels": self._taxonomy_levels(),
            "parent_of": {
                1: genus_to_family,
                2: species_to_genus,
            },
        }

        samples: List[Dict[str, Any]] = []
        for row_index, ann in enumerate(annotations):
            if not isinstance(ann, dict):
                raise TypeError(
                    f"Malformed iNat19 COCO annotation {ann_path} annotation row {row_index}: "
                    "expected a mapping."
                )
            try:
                image_id = int(ann.get("image_id"))
                species = int(ann.get("category_id"))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Malformed iNat19 COCO annotation {ann_path} annotation row {row_index}: "
                    "image_id and category_id must be integers."
                ) from exc

            image = image_by_id.get(image_id)
            category = category_by_id.get(species)
            if image is None or category is None:
                raise ValueError(
                    f"Malformed iNat19 COCO annotation {ann_path} annotation row {row_index}: "
                    f"unknown image_id={image_id} or category_id={species}."
                )

            family_key = self._normalize_taxon_name(category.get("family"))
            genus_key = self._normalize_taxon_name(category.get("genus"))
            if not family_key or not genus_key:
                raise ValueError(
                    f"Malformed iNat19 COCO annotation {ann_path} annotation row {row_index}: "
                    "referenced category is missing family or genus."
                )
            family = family_to_id.get(family_key)
            genus = genus_to_id.get((family_key, genus_key))
            if family is None or genus is None:
                raise ValueError(
                    f"Malformed iNat19 COCO annotation {ann_path} annotation row {row_index}: "
                    "referenced category is absent from the canonical family/genus maps."
                )

            file_name = str(image.get("file_name", "")).strip()
            if not file_name:
                raise ValueError(
                    f"Malformed iNat19 COCO annotation {ann_path} annotation row {row_index}: "
                    f"image id {image_id} has no file_name."
                )
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
                    raise ValueError(f"iNat19 annotation archive contains no JSON file: {path}")

                target = members[0]
                wanted_name = lower_name.replace(".tar.gz", "").replace(".tgz", ".json").replace(".tar", "")
                for member in members:
                    member_name = Path(member.name).name.lower()
                    if member_name == wanted_name:
                        target = member
                        break

                extracted = tar.extractfile(target)
                if extracted is None:
                    raise ValueError(
                        f"Could not extract iNat19 JSON member {target.name} from {path}."
                    )
                return json.loads(extracted.read().decode("utf-8"))

        raise ValueError(
            f"Unsupported iNat19 annotation format for {path}; expected .json or .tar(.gz)."
        )

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
