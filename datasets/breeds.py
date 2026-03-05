from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseHierDataset, taxonomy_from_parent_of


class BreedsDataset(BaseHierDataset):
    """BREEDS adapter aligned with H-CAST file formats."""

    def load_samples(self) -> List[Dict[str, Any]]:
        ann_file = self._annotation_file_for_split()
        if ann_file is not None:
            return self._read_json_samples(ann_file)

        sort_name = str(self.cfg.dataset.get("breeds_sort", "entity13"))
        source_train = bool(self.cfg.dataset.get("breeds_train_source", True))

        if self.split == "train":
            filename = f"{sort_name}_train_source.txt" if source_train else f"{sort_name}_train_target.txt"
        elif self.split == "val":
            filename = f"{sort_name}_val_source.txt"
        else:
            filename = f"{sort_name}_val_target.txt"

        candidates = [
            self.root / filename,
            self.root / "data" / filename,
            Path("data") / filename,
        ]

        txt_path: Optional[Path] = None
        for p in candidates:
            if p.exists():
                txt_path = p
                break

        if txt_path is None:
            return []

        samples: List[Dict[str, Any]] = []
        with txt_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                rel_path = parts[0]
                species = int(parts[1])
                coarse = int(parts[2])
                samples.append(
                    {
                        "image": self.root / rel_path,
                        "labels": [coarse, species],
                        "meta": {"source": "breeds_txt", "raw": parts},
                    }
                )

        return samples

    def load_taxonomy(self):
        tax = super().load_taxonomy()
        if tax is not None:
            return tax

        # BREEDS has 2-level hierarchy: coarse -> species.
        parent_of = {1: {int(s["labels"][1]): int(s["labels"][0]) for s in self.samples}}
        levels = list(self.cfg.dataset.get("levels", [])) or ["coarse", "species"]
        return taxonomy_from_parent_of(parent_of, levels)
