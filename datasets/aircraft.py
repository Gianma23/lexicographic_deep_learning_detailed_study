from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .base import BaseHierDataset


@dataclass(frozen=True)
class OfficialAircraftHierarchy:
    manufacturer_to_id: Dict[str, int]
    family_to_id: Dict[str, int]
    variant_to_id: Dict[str, int]
    family_to_manufacturer: Dict[int, int]
    variant_to_family: Dict[int, int]


def resolve_official_aircraft_data_root(root: Path) -> Path:
    """Resolve the ``data`` directory of a complete official Aircraft download."""
    root = Path(root)
    candidates = [
        root / "fgvc-aircraft-2013b" / "data",
        root / "data",
        root,
    ]
    required = {
        "variants.txt",
        "families.txt",
        "manufacturers.txt",
        "images_variant_train.txt",
        "images_variant_val.txt",
        "images_variant_test.txt",
    }

    for candidate in candidates:
        if not candidate.exists() or not (candidate / "images").is_dir():
            continue
        if all((candidate / filename).is_file() for filename in required):
            return candidate

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not locate a complete official FGVC-Aircraft download. "
        "Expected the data directory to contain images/, variants.txt, families.txt, "
        "manufacturers.txt, and the official train/val/test annotation files. "
        f"Searched: {searched}"
    )


class AircraftDataset(BaseHierDataset):
    """FGVC-Aircraft adapter built only from the official download annotations."""

    _OFFICIAL_SPLITS = ("train", "val", "test")

    def default_levels(self) -> List[str]:
        """Default hierarchy names used when config does not provide levels."""
        return ["manufacturer", "family", "variant"]

    def load_samples(self) -> List[Dict[str, Any]]:
        """Load one official split and derive its labels from the official hierarchy."""
        if self.split not in self._OFFICIAL_SPLITS:
            raise ValueError(
                f"Unsupported FGVC-Aircraft split '{self.split}'. "
                f"Expected one of {list(self._OFFICIAL_SPLITS)}."
            )

        data_root = self._resolve_data_root()
        hierarchy = self._build_official_hierarchy(data_root)
        self._official_hierarchy = hierarchy

        labels_path = data_root / f"images_variant_{self.split}.txt"
        if not labels_path.exists():
            raise FileNotFoundError(
                f"Missing official FGVC-Aircraft split annotation: {labels_path}"
            )

        samples = self._read_variant_file(
            labels_path=labels_path,
            images_dir=data_root / "images",
            hierarchy=hierarchy,
            source=f"aircraft_{self.split}",
        )
        self._validate_split_class_coverage(
            samples=samples,
            hierarchy=hierarchy,
            split=self.split,
        )
        return samples

    def load_taxonomy(self) -> Dict[str, Any]:
        """Return the complete official taxonomy, independent of the active split."""
        hierarchy = getattr(self, "_official_hierarchy", None)
        if hierarchy is None:
            raise RuntimeError("FGVC-Aircraft hierarchy was not initialized before taxonomy loading.")
        return {
            "levels": self._taxonomy_levels(),
            "parent_of": {
                1: dict(hierarchy.family_to_manufacturer),
                2: dict(hierarchy.variant_to_family),
            },
        }

    def _resolve_data_root(self) -> Path:
        """Resolve the official download's ``data`` directory."""
        return resolve_official_aircraft_data_root(self.root)

    @staticmethod
    def _read_class_names(path: Path, level_name: str) -> Dict[str, int]:
        """Read an official class list while preserving its published order."""
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing official FGVC-Aircraft {level_name} class list: {path}"
            )

        names: Dict[str, int] = {}
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                name = line.strip()
                if not name:
                    continue
                if name in names:
                    raise ValueError(
                        f"Duplicate {level_name} name '{name}' in {path} at line {line_number}."
                    )
                names[name] = len(names)

        if not names:
            raise ValueError(f"Official FGVC-Aircraft {level_name} class list is empty: {path}")
        return names

    @staticmethod
    def _read_image_labels(path: Path) -> Dict[str, str]:
        """Read an official ``image_id class name`` annotation file."""
        if not path.is_file():
            raise FileNotFoundError(f"Missing official FGVC-Aircraft annotation file: {path}")

        labels: Dict[str, str] = {}
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = line.strip().split(" ", 1)
                if len(row) != 2 or not row[0].strip() or not row[1].strip():
                    raise ValueError(
                        f"Malformed FGVC-Aircraft annotation in {path} at line {line_number}: "
                        f"{line.rstrip()!r}"
                    )
                image_id, class_name = row[0].strip(), row[1].strip()
                previous = labels.get(image_id)
                if previous is not None and previous != class_name:
                    raise ValueError(
                        f"Conflicting labels for image '{image_id}' in {path}: "
                        f"'{previous}' versus '{class_name}'."
                    )
                labels[image_id] = class_name

        if not labels:
            raise ValueError(f"Official FGVC-Aircraft annotation file is empty: {path}")
        return labels

    @classmethod
    def _build_official_hierarchy(cls, data_root: Path) -> OfficialAircraftHierarchy:
        """Join official per-image annotations into variant and family parent maps."""
        manufacturer_to_id = cls._read_class_names(
            data_root / "manufacturers.txt",
            level_name="manufacturer",
        )
        family_to_id = cls._read_class_names(
            data_root / "families.txt",
            level_name="family",
        )
        variant_to_id = cls._read_class_names(
            data_root / "variants.txt",
            level_name="variant",
        )

        family_to_manufacturer: Dict[int, int] = {}
        variant_to_family: Dict[int, int] = {}

        for split in cls._OFFICIAL_SPLITS:
            variant_path = data_root / f"images_variant_{split}.txt"
            family_path = data_root / f"images_family_{split}.txt"
            manufacturer_path = data_root / f"images_manufacturer_{split}.txt"

            variants = cls._read_image_labels(variant_path)
            families = cls._read_image_labels(family_path)
            manufacturers = cls._read_image_labels(manufacturer_path)

            variant_images = set(variants)
            family_images = set(families)
            manufacturer_images = set(manufacturers)
            if variant_images != family_images or variant_images != manufacturer_images:
                raise ValueError(
                    f"Official FGVC-Aircraft {split} annotations do not contain identical image IDs: "
                    f"variant={len(variant_images)}, family={len(family_images)}, "
                    f"manufacturer={len(manufacturer_images)}."
                )

            for image_id in sorted(variant_images):
                variant_name = variants[image_id]
                family_name = families[image_id]
                manufacturer_name = manufacturers[image_id]
                try:
                    variant_id = variant_to_id[variant_name]
                except KeyError as exc:
                    raise ValueError(
                        f"Unknown variant '{variant_name}' for image '{image_id}' in {variant_path}."
                    ) from exc
                try:
                    family_id = family_to_id[family_name]
                except KeyError as exc:
                    raise ValueError(
                        f"Unknown family '{family_name}' for image '{image_id}' in {family_path}."
                    ) from exc
                try:
                    manufacturer_id = manufacturer_to_id[manufacturer_name]
                except KeyError as exc:
                    raise ValueError(
                        f"Unknown manufacturer '{manufacturer_name}' for image '{image_id}' "
                        f"in {manufacturer_path}."
                    ) from exc

                cls._record_unique_parent(
                    mapping=variant_to_family,
                    child_id=variant_id,
                    parent_id=family_id,
                    child_name=variant_name,
                    parent_name=family_name,
                    relation="variant-to-family",
                )
                cls._record_unique_parent(
                    mapping=family_to_manufacturer,
                    child_id=family_id,
                    parent_id=manufacturer_id,
                    child_name=family_name,
                    parent_name=manufacturer_name,
                    relation="family-to-manufacturer",
                )

        cls._require_complete_mapping(
            mapping=variant_to_family,
            class_names=variant_to_id,
            relation="variant-to-family",
        )
        cls._require_complete_mapping(
            mapping=family_to_manufacturer,
            class_names=family_to_id,
            relation="family-to-manufacturer",
        )

        return OfficialAircraftHierarchy(
            manufacturer_to_id=manufacturer_to_id,
            family_to_id=family_to_id,
            variant_to_id=variant_to_id,
            family_to_manufacturer=family_to_manufacturer,
            variant_to_family=variant_to_family,
        )

    @staticmethod
    def _record_unique_parent(
        mapping: Dict[int, int],
        child_id: int,
        parent_id: int,
        child_name: str,
        parent_name: str,
        relation: str,
    ) -> None:
        previous = mapping.get(child_id)
        if previous is not None and previous != parent_id:
            raise ValueError(
                f"Inconsistent official FGVC-Aircraft {relation} mapping for '{child_name}': "
                f"parent id {previous} versus '{parent_name}' (id {parent_id})."
            )
        mapping[child_id] = parent_id

    @staticmethod
    def _require_complete_mapping(
        mapping: Dict[int, int],
        class_names: Dict[str, int],
        relation: str,
    ) -> None:
        id_to_name = {class_id: name for name, class_id in class_names.items()}
        missing_ids = sorted(set(id_to_name) - set(mapping))
        if missing_ids:
            missing_names = [id_to_name[class_id] for class_id in missing_ids[:10]]
            raise ValueError(
                f"Incomplete official FGVC-Aircraft {relation} mapping. "
                f"Missing {len(missing_ids)} classes, including: {missing_names}."
            )

    @staticmethod
    def _read_variant_file(
        labels_path: Path,
        images_dir: Path,
        hierarchy: OfficialAircraftHierarchy,
        source: str,
    ) -> List[Dict[str, Any]]:
        """Emit normalized samples using official variant and parent IDs."""
        samples: List[Dict[str, Any]] = []
        for image_id, variant_name in AircraftDataset._read_image_labels(labels_path).items():
            try:
                variant_id = hierarchy.variant_to_id[variant_name]
            except KeyError as exc:
                raise ValueError(
                    f"Unknown variant '{variant_name}' for image '{image_id}' in {labels_path}."
                ) from exc

            family_id = hierarchy.variant_to_family[variant_id]
            manufacturer_id = hierarchy.family_to_manufacturer[family_id]
            image_file = image_id if Path(image_id).suffix else f"{image_id}.jpg"
            samples.append(
                {
                    "image": images_dir / image_file,
                    "labels": [manufacturer_id, family_id, variant_id],
                    "meta": {
                        "source": source,
                        "manufacturer_name": next(
                            name
                            for name, class_id in hierarchy.manufacturer_to_id.items()
                            if class_id == manufacturer_id
                        ),
                        "family_name": next(
                            name
                            for name, class_id in hierarchy.family_to_id.items()
                            if class_id == family_id
                        ),
                        "variant_name": variant_name,
                        "labels_file": str(labels_path),
                    },
                }
            )
        return samples

    @staticmethod
    def _validate_split_class_coverage(
        samples: List[Dict[str, Any]],
        hierarchy: OfficialAircraftHierarchy,
        split: str,
    ) -> None:
        """Prevent split-local remapping from changing official class IDs."""
        expected_per_level = [
            set(hierarchy.manufacturer_to_id.values()),
            set(hierarchy.family_to_id.values()),
            set(hierarchy.variant_to_id.values()),
        ]
        observed_per_level = [
            {int(sample["labels"][level]) for sample in samples}
            for level in range(3)
        ]
        level_names = ("manufacturer", "family", "variant")
        for level_name, expected, observed in zip(
            level_names,
            expected_per_level,
            observed_per_level,
        ):
            if observed != expected:
                missing = sorted(expected - observed)
                extra = sorted(observed - expected)
                raise ValueError(
                    f"Official FGVC-Aircraft {split} split does not cover the complete "
                    f"{level_name} class list. Missing ids: {missing[:10]}, "
                    f"extra ids: {extra[:10]}."
                )


def load_official_aircraft_hierarchy(root: Path) -> OfficialAircraftHierarchy:
    """Build the complete hierarchy directly from an official Aircraft download."""
    data_root = resolve_official_aircraft_data_root(root)
    return AircraftDataset._build_official_hierarchy(data_root)
