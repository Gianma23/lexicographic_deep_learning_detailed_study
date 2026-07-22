import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .base import BaseHierDataset
from .cub_tree import TREES
from .splitting import split_train_val_samples


class CUBDataset(BaseHierDataset):
    """CUB-200 adapter with H-CAST order/family/species hierarchy."""

    def default_levels(self) -> List[str]:
        """Default hierarchy names used when config does not provide levels."""
        return ["order", "family", "species"]

    def load_samples(self) -> List[Dict[str, Any]]:
        """Load split samples from JSON, split folders, or official CUB files."""
        ann_file = self._annotation_file_for_split()
        if ann_file is not None:
            return self._read_json_samples(ann_file)

        train_samples, test_samples = self._load_from_split_folders()
        if not train_samples and not test_samples:
            train_samples, test_samples = self._load_from_official_files()

        if self.split == "test":
            return test_samples

        return split_train_val_samples(train_samples, split=self.split, cfg=self.cfg, stratify_level=-1)

    def _load_from_split_folders(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Read train/test samples from common folder-based split layouts."""
        candidates = [
            (self.root / "train", self.root / "test", "cub_folder"),
            (self.root / "images_split" / "train", self.root / "images_split" / "test", "cub_images_split"),
        ]

        for train_dir, test_dir, source in candidates:
            if not train_dir.exists() or not test_dir.exists():
                continue
            train_samples = self._read_folder_classes(train_dir, source=f"{source}_train")
            test_samples = self._read_folder_classes(test_dir, source=f"{source}_test")
            return train_samples, test_samples

        return [], []

    def _read_folder_classes(self, root_dir: Path, source: str) -> List[Dict[str, Any]]:
        """Parse class subfolders and map images to order/family/species labels."""
        class_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])
        fallback_species = {p.name: i for i, p in enumerate(class_dirs)}

        samples: List[Dict[str, Any]] = []
        for class_dir in class_dirs:
            species = self._species_from_class_name(class_dir.name, fallback_species[class_dir.name])
            if species < 0 or species >= len(TREES):
                raise ValueError(
                    f"CUB class folder {class_dir} resolves to out-of-range species id {species}."
                )

            tree = TREES[species]
            order = int(tree[1]) - 1
            family = int(tree[2]) - 1
            for img in class_dir.rglob("*"):
                if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                samples.append(
                    {
                        "image": img,
                        "labels": [order, family, species],
                        "meta": {"source": source, "class": class_dir.name},
                    }
                )

        return samples

    @staticmethod
    def _species_from_class_name(class_name: str, fallback: int) -> int:
        """Extract species ID from a class folder prefix, or use a fallback ID."""
        match = re.match(r"^(\d+)", class_name)
        if match:
            species = int(match.group(1)) - 1
            if species >= 0:
                return species
        return fallback

    def _load_from_official_files(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load CUB-200 from official metadata files and split flags."""
        roots = [self.root, self.root / "CUB_200_2011"]
        for base in roots:
            images_txt = base / "images.txt"
            labels_txt = base / "image_class_labels.txt"
            split_txt = base / "train_test_split.txt"
            images_dir = base / "images"

            if not (images_txt.exists() and labels_txt.exists() and split_txt.exists() and images_dir.exists()):
                continue

            image_map = self._read_int_str_map(images_txt)
            class_map = self._read_int_int_map(labels_txt)
            split_map = self._read_int_int_map(split_txt)
            image_ids = set(image_map)
            if set(class_map) != image_ids or set(split_map) != image_ids:
                raise ValueError(
                    f"Malformed CUB metadata under {base}: images.txt, "
                    "image_class_labels.txt, and train_test_split.txt must contain "
                    "identical image ids."
                )

            train_samples: List[Dict[str, Any]] = []
            test_samples: List[Dict[str, Any]] = []
            for image_id in sorted(image_map.keys()):
                if image_id not in class_map or image_id not in split_map:
                    raise ValueError(
                        f"Malformed CUB metadata under {base}: image id {image_id} from "
                        "images.txt is missing from image_class_labels.txt or train_test_split.txt."
                    )

                species = int(class_map[image_id]) - 1
                if species < 0 or species >= len(TREES):
                    raise ValueError(
                        f"Malformed CUB metadata under {base}: image id {image_id} has "
                        f"out-of-range one-based species id {class_map[image_id]}."
                    )
                split_flag = int(split_map[image_id])
                if split_flag not in {0, 1}:
                    raise ValueError(
                        f"Malformed CUB metadata under {base}: image id {image_id} has "
                        f"invalid train/test flag {split_flag}; expected 0 or 1."
                    )

                tree = TREES[species]
                order = int(tree[1]) - 1
                family = int(tree[2]) - 1

                rel = image_map[image_id]
                image_path = images_dir / rel
                if not image_path.exists():
                    image_path = base / rel

                sample = {
                    "image": image_path,
                    "labels": [order, family, species],
                    "meta": {"source": "cub_official", "image_id": image_id},
                }

                if split_flag == 1:
                    train_samples.append(sample)
                else:
                    test_samples.append(sample)

            return train_samples, test_samples

        return [], []

    @staticmethod
    def _read_int_str_map(path: Path) -> Dict[int, str]:
        """Read two-column text files mapping integer keys to string values."""
        out: Dict[int, str] = {}
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                row = line.strip().split(maxsplit=1)
                if len(row) != 2:
                    raise ValueError(
                        f"Malformed CUB metadata in {path} at line {line_number}: "
                        f"expected '<integer id> <value>', got {line.rstrip()!r}."
                    )
                try:
                    key = int(row[0])
                except ValueError as exc:
                    raise ValueError(
                        f"Malformed CUB metadata in {path} at line {line_number}: "
                        f"id {row[0]!r} is not an integer."
                    ) from exc
                if key in out:
                    raise ValueError(
                        f"Malformed CUB metadata in {path} at line {line_number}: "
                        f"duplicate id {key}."
                    )
                out[key] = row[1]
        if not out:
            raise ValueError(f"CUB metadata file is empty: {path}")
        return out

    @staticmethod
    def _read_int_int_map(path: Path) -> Dict[int, int]:
        """Read two-column text files mapping integer keys to integer values."""
        out: Dict[int, int] = {}
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                row = line.strip().split(maxsplit=1)
                if len(row) != 2:
                    raise ValueError(
                        f"Malformed CUB metadata in {path} at line {line_number}: "
                        f"expected two integers, got {line.rstrip()!r}."
                    )
                try:
                    key = int(row[0])
                    value = int(row[1])
                except ValueError as exc:
                    raise ValueError(
                        f"Malformed CUB metadata in {path} at line {line_number}: "
                        f"expected two integers, got {line.rstrip()!r}."
                    ) from exc
                if key in out:
                    raise ValueError(
                        f"Malformed CUB metadata in {path} at line {line_number}: "
                        f"duplicate id {key}."
                    )
                out[key] = value
        if not out:
            raise ValueError(f"CUB metadata file is empty: {path}")
        return out
