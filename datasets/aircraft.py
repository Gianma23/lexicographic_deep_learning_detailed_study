import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from .aircraft_tree import TREES
from .base import BaseHierDataset, taxonomy_from_parent_of


class AircraftDataset(BaseHierDataset):
    """FGVC-Aircraft adapter aligned with H-CAST hierarchy."""

    def load_samples(self) -> List[Dict[str, Any]]:
        ann_file = self._annotation_file_for_split()
        if ann_file is not None:
            return self._read_json_samples(ann_file)

        data_root = self.root / "fgvc-aircraft-2013b" / "data"
        if not data_root.exists():
            return []

        split_file = "images_variant_trainval.txt" if self.split in {"train", "val"} else "images_variant_test.txt"
        labels_path = data_root / split_file
        images_dir = data_root / "images"

        if not labels_path.exists():
            return []

        # H-CAST maps variant label names to ids using Air.csv.
        air_csv_candidates = [self.root / "Air.csv", self.root / "data" / "Air.csv", Path("data") / "Air.csv"]
        csv_path: Optional[Path] = None
        for p in air_csv_candidates:
            if p.exists():
                csv_path = p
                break

        variant_to_id: Dict[str, int] = {}
        if csv_path is not None:
            with csv_path.open("r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 2:
                        continue
                    variant_name = row[1].strip()
                    try:
                        variant_to_id[variant_name] = int(row[0].strip()) - 1
                    except ValueError:
                        continue

        samples: List[Dict[str, Any]] = []
        with labels_path.open("r", encoding="utf-8") as f:
            for line in f:
                image_name, variant_name = line.strip().split(" ", 1)
                if variant_name not in variant_to_id:
                    continue
                variant = int(variant_to_id[variant_name])
                tree = TREES[variant]
                family = int(tree[1]) - 1
                manufacturer = int(tree[2]) - 1
                samples.append(
                    {
                        "image": images_dir / f"{image_name}.jpg",
                        "labels": [manufacturer, family, variant],
                        "meta": {"source": "aircraft_txt", "variant_name": variant_name},
                    }
                )

        return samples

    def load_taxonomy(self):
        tax = super().load_taxonomy()
        if tax is not None:
            return tax

        parent_of = {
            1: {int(s["labels"][1]): int(s["labels"][0]) for s in self.samples},
            2: {int(s["labels"][2]): int(s["labels"][1]) for s in self.samples},
        }
        levels = list(self.cfg.dataset.get("levels", [])) or ["manufacturer", "family", "variant"]
        return taxonomy_from_parent_of(parent_of, levels)
