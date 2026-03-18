import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .aircraft_tree import TREES
from .base import BaseHierDataset, split_train_val_samples


class AircraftDataset(BaseHierDataset):
    """FGVC-Aircraft adapter aligned with H-CAST hierarchy."""

    def default_levels(self) -> List[str]:
        """Default hierarchy names used when config does not provide levels."""
        return ["manufacturer", "family", "variant"]

    def load_samples(self) -> List[Dict[str, Any]]:
        """Load split samples from JSON or FGVC-Aircraft text annotations."""
        ann_file = self._annotation_file_for_split()
        if ann_file is not None:
            return self._read_json_samples(ann_file)

        data_root = self._resolve_data_root()
        images_dir = data_root / "images"

        train_file = data_root / "images_variant_train.txt"
        val_file = data_root / "images_variant_val.txt"
        test_file = data_root / "images_variant_test.txt"
        trainval_file = data_root / "images_variant_trainval.txt"

        variant_to_id = self._build_variant_to_id(data_root)

        if self.split == "test":
            if test_file.exists():
                return self._read_variant_file(test_file, images_dir, variant_to_id, source="aircraft_test")
            return []

        if train_file.exists() and val_file.exists():
            if self.split == "train":
                return self._read_variant_file(train_file, images_dir, variant_to_id, source="aircraft_train")
            return self._read_variant_file(val_file, images_dir, variant_to_id, source="aircraft_val")

        if train_file.exists():
            pool = self._read_variant_file(train_file, images_dir, variant_to_id, source="aircraft_train")
            return split_train_val_samples(pool, split=self.split, cfg=self.cfg, stratify_level=-1)

        if trainval_file.exists():
            pool = self._read_variant_file(trainval_file, images_dir, variant_to_id, source="aircraft_trainval")
            return split_train_val_samples(pool, split=self.split, cfg=self.cfg, stratify_level=-1)

        return []

    def _resolve_data_root(self) -> Path:
        """Resolve the directory that contains images and split label files."""
        candidates = [
            self.root / "fgvc-aircraft-2013b" / "data",
            self.root / "data",
            self.root,
        ]
        split_files = {
            "images_variant_train.txt",
            "images_variant_val.txt",
            "images_variant_test.txt",
            "images_variant_trainval.txt",
        }

        for candidate in candidates:
            if not candidate.exists():
                continue
            if (candidate / "images").exists() and any((candidate / f).exists() for f in split_files):
                return candidate

        return candidates[0]

    def _build_variant_to_id(self, data_root: Path) -> Dict[str, int]:
        """Build variant-name to index mapping from CSV, variants.txt, or splits."""
        air_csv_candidates = [
            self.root / "Air.csv",
            self.root / "data" / "Air.csv",
            data_root / "Air.csv",
            Path("data") / "Air.csv",
        ]
        for csv_path in air_csv_candidates:
            if not csv_path.exists():
                continue

            variant_to_id: Dict[str, int] = {}
            with csv_path.open("r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 2:
                        continue
                    try:
                        idx = int(row[0].strip()) - 1
                    except ValueError:
                        continue
                    name = row[1].strip()
                    if name:
                        variant_to_id[name] = idx
            if variant_to_id:
                return variant_to_id

        variants_txt = data_root / "variants.txt"
        if variants_txt.exists():
            out: Dict[str, int] = {}
            with variants_txt.open("r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    name = line.strip()
                    if name:
                        out[name] = idx
            if out:
                return out

        names: Set[str] = set()
        for filename in [
            "images_variant_train.txt",
            "images_variant_val.txt",
            "images_variant_test.txt",
            "images_variant_trainval.txt",
        ]:
            path = data_root / filename
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    row = line.strip().split(" ", 1)
                    if len(row) == 2 and row[1].strip():
                        names.add(row[1].strip())

        return {name: idx for idx, name in enumerate(sorted(names))}

    def _read_variant_file(
        self,
        labels_path: Path,
        images_dir: Path,
        variant_to_id: Dict[str, int],
        source: str,
    ) -> List[Dict[str, Any]]:
        """Parse FGVC variant label files and emit normalized sample dicts."""
        samples: List[Dict[str, Any]] = []
        with labels_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = line.strip().split(" ", 1)
                if len(row) != 2:
                    continue
                image_name, variant_name = row[0].strip(), row[1].strip()
                if not image_name or not variant_name:
                    continue

                variant = variant_to_id.get(variant_name)
                if variant is None or variant < 0 or variant >= len(TREES):
                    continue

                tree = TREES[variant]
                family = int(tree[1]) - 1
                manufacturer = int(tree[2]) - 1

                image_file = image_name if Path(image_name).suffix else f"{image_name}.jpg"
                image_path = images_dir / image_file
                samples.append(
                    {
                        "image": image_path,
                        "labels": [manufacturer, family, variant],
                        "meta": {
                            "source": source,
                            "variant_name": variant_name,
                            "labels_file": str(labels_path),
                        },
                    }
                )

        return samples
