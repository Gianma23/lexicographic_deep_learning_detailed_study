#!/usr/bin/env python3
"""Create fixed iNat19 manifests from Making Better Mistakes split files."""

import argparse
import json
import os
import sys
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402


load_dotenv(
    Path(os.environ.get("PROJECT_ENV_FILE", REPO_ROOT / ".env")).expanduser(),
    override=False,
)


EXPECTED_MBM_COUNTS = {
    "train": 187385,
    "val": 40121,
    "test": 40737,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert official iNaturalist 2019 train/val annotations plus "
            "Making Better Mistakes splits_inat19.zip into repo-normalized "
            "train/val/test manifests."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("INAT19_ROOT", "/scratch/g.saggini1/datasets/inat19")),
        help="iNat19 root containing train_val2019/, train2019.json, val2019.json, and splits_inat19.zip.",
    )
    parser.add_argument(
        "--splits-zip",
        type=Path,
        default=None,
        help="Path to splits_inat19.zip. Defaults to <root>/splits_inat19.zip.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for train.json/val.json/test.json. Defaults to <root>/annotations_mbm.",
    )
    return parser.parse_args()


def load_json_payload(path: Path) -> Any:
    """Load JSON from a plain file or from a JSON-in-tar archive."""
    lower_name = path.name.lower()
    if lower_name.endswith(".json"):
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    if lower_name.endswith(".tar.gz") or lower_name.endswith(".tgz") or lower_name.endswith(".tar"):
        with tarfile.open(path, "r:*") as tar:
            members = [m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith(".json")]
            if not members:
                raise FileNotFoundError(f"No JSON member found in archive: {path}")

            target = members[0]
            wanted_name = lower_name.replace(".tar.gz", "").replace(".tgz", ".json").replace(".tar", "")
            for member in members:
                if Path(member.name).name.lower() == wanted_name:
                    target = member
                    break

            extracted = tar.extractfile(target)
            if extracted is None:
                raise FileNotFoundError(f"Could not extract JSON member {target.name} from {path}")
            return json.loads(extracted.read().decode("utf-8"))

    raise ValueError(f"Unsupported annotation file type: {path}")


def resolve_annotation(root: Path, stem: str) -> Path:
    """Find an extracted or compressed official annotation file."""
    candidates = [
        root / f"{stem}.json",
        root / f"{stem}.json.tar.gz",
        root / f"{stem}.json.tgz",
        root / f"{stem}.json.tar",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {stem}.json(.tar.gz) under {root}")


def normalize_taxon_name(value: Any) -> str:
    return str(value or "").strip().lower()


def build_category_maps(payloads: Iterable[Dict[str, Any]]) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, int], Dict[Tuple[str, str], int]]:
    """Build stable family/genus mappings across official train and val annotations."""
    category_by_id: Dict[int, Dict[str, Any]] = {}
    family_keys = set()
    genus_keys = set()

    for payload in payloads:
        categories = payload.get("categories", [])
        if not isinstance(categories, list):
            continue

        for category in categories:
            if not isinstance(category, dict):
                continue
            try:
                category_id = int(category.get("id"))
            except (TypeError, ValueError):
                continue

            category_by_id.setdefault(category_id, category)
            family_key = normalize_taxon_name(category.get("family"))
            genus_key = normalize_taxon_name(category.get("genus"))
            if family_key and genus_key:
                family_keys.add(family_key)
                genus_keys.add((family_key, genus_key))

    family_to_id = {name: idx for idx, name in enumerate(sorted(family_keys))}
    genus_to_id = {name: idx for idx, name in enumerate(sorted(genus_keys))}
    return category_by_id, family_to_id, genus_to_id


def candidate_image_paths(root: Path, file_name: str, category_dir: str) -> List[Path]:
    """Return likely absolute image paths for an official iNat19 image row."""
    rel = Path(file_name)
    basename = rel.name
    candidates = [
        root / rel,
        root / "train_val2019" / rel,
    ]
    if category_dir and basename:
        cat_rel = Path(category_dir) / basename
        candidates.extend(
            [
                root / cat_rel,
                root / "train_val2019" / cat_rel,
            ]
        )

    deduped = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def resolve_image_relpath(root: Path, file_name: str, category_dir: str) -> str:
    """Resolve an official image row to a root-relative path."""
    candidates = candidate_image_paths(root, file_name=file_name, category_dir=category_dir)
    for candidate in candidates:
        if candidate.exists():
            return candidate.relative_to(root).as_posix()

    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not resolve image for file_name={file_name!r}. Tried: {tried}")


def index_official_annotations(
    root: Path,
    payloads_by_split: Dict[str, Dict[str, Any]],
    category_by_id: Dict[int, Dict[str, Any]],
    family_to_id: Dict[str, int],
    genus_to_id: Dict[Tuple[str, str], int],
) -> Dict[str, List[Dict[str, Any]]]:
    """Index official train/val annotations by image basename."""
    by_basename: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for official_split, payload in payloads_by_split.items():
        images = payload.get("images", [])
        annotations = payload.get("annotations", [])
        if not isinstance(images, list) or not isinstance(annotations, list):
            raise ValueError(f"{official_split} annotation payload is not COCO-style.")

        image_by_id: Dict[int, Dict[str, Any]] = {}
        for image in images:
            if not isinstance(image, dict):
                continue
            try:
                image_id = int(image.get("id"))
            except (TypeError, ValueError):
                continue
            image_by_id[image_id] = image

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

            family_key = normalize_taxon_name(category.get("family"))
            genus_key = normalize_taxon_name(category.get("genus"))
            if family_key not in family_to_id or (family_key, genus_key) not in genus_to_id:
                continue

            file_name = str(image.get("file_name", "")).strip()
            if not file_name:
                continue

            category_dir = str(category.get("image_dir_name", "")).strip()
            image_rel = resolve_image_relpath(root, file_name=file_name, category_dir=category_dir)
            basename = Path(file_name).name
            by_basename[basename].append(
                {
                    "image": image_rel,
                    "labels": [family_to_id[family_key], genus_to_id[(family_key, genus_key)], species],
                    "meta": {
                        "source": "inat19_mbm_split",
                        "official_split": official_split,
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

    return by_basename


def split_entries(splits_zip: Path) -> Dict[str, List[Tuple[str, str]]]:
    """Read MBM split entries as (nat_class, basename) tuples by split."""
    entries: Dict[str, List[Tuple[str, str]]] = {split: [] for split in EXPECTED_MBM_COUNTS}
    with zipfile.ZipFile(splits_zip) as zf:
        for name in sorted(zf.namelist()):
            parts = name.split("/")
            if len(parts) != 3 or not parts[2].endswith(".txt"):
                continue
            _, split, filename = parts
            if split not in entries:
                continue
            nat_class = Path(filename).stem
            lines = zf.read(name).decode("utf-8").splitlines()
            for line in lines:
                basename = line.strip()
                if basename:
                    entries[split].append((nat_class, basename))
    return entries


def build_split_samples(
    entries: Dict[str, List[Tuple[str, str]]],
    official_by_basename: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Join MBM split rows with official labels and image paths."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    missing = []
    ambiguous = []

    for split, rows in entries.items():
        samples = []
        for nat_class, basename in rows:
            matches = official_by_basename.get(basename, [])
            if not matches:
                missing.append((split, nat_class, basename))
                continue
            if len(matches) > 1:
                ambiguous.append((split, nat_class, basename, len(matches)))
                continue

            sample = dict(matches[0])
            meta = dict(sample.get("meta", {}))
            meta["mbm_split"] = split
            meta["mbm_class"] = nat_class
            sample["meta"] = meta
            samples.append(sample)
        out[split] = samples

    if missing or ambiguous:
        messages = []
        if missing:
            preview = "; ".join(f"{split}/{cls}/{name}" for split, cls, name in missing[:8])
            messages.append(f"{len(missing)} MBM entries were missing from official annotations: {preview}")
        if ambiguous:
            preview = "; ".join(f"{split}/{cls}/{name} ({count})" for split, cls, name, count in ambiguous[:8])
            messages.append(f"{len(ambiguous)} MBM entries matched multiple official images: {preview}")
        raise RuntimeError(" ".join(messages))

    return out


def write_manifests(output_dir: Path, samples_by_split: Dict[str, List[Dict[str, Any]]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        path = output_dir / f"{split}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump({"samples": samples_by_split[split]}, f, separators=(",", ":"))
            f.write("\n")


def print_counts(samples_by_split: Dict[str, List[Dict[str, Any]]]) -> None:
    print("iNat19 MBM manifest counts:")
    for split in ["train", "val", "test"]:
        actual = len(samples_by_split.get(split, []))
        expected = EXPECTED_MBM_COUNTS[split]
        status = "OK" if actual == expected else "CHECK"
        print(f"  {split}: {actual} (expected MBM full split: {expected}) [{status}]")

    species_counts = Counter()
    for samples in samples_by_split.values():
        for sample in samples:
            labels = sample.get("labels", [])
            if len(labels) == 3:
                species_counts[int(labels[2])] += 1
    print(f"  species with at least one image: {len(species_counts)}")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    splits_zip = (args.splits_zip or root / "splits_inat19.zip").expanduser().resolve()
    output_dir = (args.output_dir or root / "annotations_mbm").expanduser().resolve()

    if not (root / "train_val2019").exists():
        raise FileNotFoundError(f"Expected image directory not found: {root / 'train_val2019'}")
    if not splits_zip.exists():
        raise FileNotFoundError(f"Expected MBM split archive not found: {splits_zip}")

    train_ann = resolve_annotation(root, "train2019")
    val_ann = resolve_annotation(root, "val2019")
    payloads_by_split = {
        "train2019": load_json_payload(train_ann),
        "val2019": load_json_payload(val_ann),
    }

    category_by_id, family_to_id, genus_to_id = build_category_maps(payloads_by_split.values())
    official_by_basename = index_official_annotations(
        root=root,
        payloads_by_split=payloads_by_split,
        category_by_id=category_by_id,
        family_to_id=family_to_id,
        genus_to_id=genus_to_id,
    )
    entries = split_entries(splits_zip)
    samples_by_split = build_split_samples(entries, official_by_basename)

    write_manifests(output_dir, samples_by_split)
    print_counts(samples_by_split)
    print(f"Wrote manifests to: {output_dir}")


if __name__ == "__main__":
    main()
