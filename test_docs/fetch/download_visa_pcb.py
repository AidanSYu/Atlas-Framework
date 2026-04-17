"""Download a small PCB1 fixture subset from the VisA anomaly-detection dataset.

Target layout (relative to the repo root)::

    test_docs/pcb/reference/good_000.jpg ... good_009.jpg   (10 normal boards)
    test_docs/pcb/defects/defect_0.jpg ... defect_4.jpg     (5 defective boards)

Source
------
The VisA (Visual Anomaly) dataset was released by Amazon Science with the
paper "SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection
and Segmentation" (ECCV 2022). Canonical repo:

    https://github.com/amazon-science/spot-diff

We pull from the Hugging Face mirror ``BrachioLab/visa`` which hosts the same
data as parquet shards (``data/pcb1.train-*`` holds normal-only training
images; ``data/pcb1.test-*`` is a 100/100 normal/defect split).

License
-------
VisA is distributed under the **Creative Commons Attribution-ShareAlike 4.0
International License (CC BY-SA 4.0)**. See:

    https://creativecommons.org/licenses/by-sa/4.0/

Any redistribution of the fixture images must retain attribution to Amazon
Science and remain under CC BY-SA 4.0.

Usage
-----
    python test_docs/fetch/download_visa_pcb.py

The script is idempotent: existing files are left alone, missing ones are
re-fetched from the cached parquet shards. Parquet shards are cached under
``test_docs/fetch/cache/`` (gitignored) so re-runs are offline-friendly.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

REPO_ID = "BrachioLab/visa"
PCB1_TRAIN = "data/pcb1.train-00000-of-00001.parquet"  # normal-only
PCB1_TEST = "data/pcb1.test-00000-of-00001.parquet"    # 100 normal + 100 defect

# How many fixtures we keep (keeps the git-committed footprint small).
N_REFERENCE = 10
N_DEFECTS = 5

# Resize long side to this to keep each committed JPEG tiny while staying
# well above the 256x256 that PatchCore consumes internally.
RESIZE_LONG = 512
JPEG_QUALITY = 88

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PCB_ROOT = REPO_ROOT / "test_docs" / "pcb"
REF_DIR = PCB_ROOT / "reference"
DEF_DIR = PCB_ROOT / "defects"
CACHE_DIR = HERE / "cache"


def _expected_paths() -> tuple[list[Path], list[Path]]:
    ref = [REF_DIR / f"good_{i:03d}.jpg" for i in range(N_REFERENCE)]
    defects = [DEF_DIR / f"defect_{i}.jpg" for i in range(N_DEFECTS)]
    return ref, defects


def _all_present() -> bool:
    ref, defects = _expected_paths()
    return all(p.is_file() for p in ref + defects)


def _require(pkg: str):
    try:
        return __import__(pkg)
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency '{pkg}'. Install with: pip install {pkg}"
        ) from exc


def _fetch_shard(filename: str) -> Path:
    hub = _require("huggingface_hub")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = hub.hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        cache_dir=str(CACHE_DIR),
    )
    return Path(path)


def _iter_rows(shard_path: Path):
    pd = _require("pandas")
    df = pd.read_parquet(shard_path)
    required = {"image", "label"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Parquet shard {shard_path} missing expected columns {missing}."
        )
    return df


def _extract_bytes(cell) -> bytes:
    if isinstance(cell, dict):
        data = cell.get("bytes")
        if data is None:
            raise RuntimeError("Image cell dict has no 'bytes' entry.")
        return data
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    raise RuntimeError(f"Unrecognized image cell type: {type(cell)!r}")


def _save_resized(img_bytes: bytes, dest: Path) -> None:
    PIL = _require("PIL")
    from PIL import Image  # noqa: WPS433 -- after _require ensures availability

    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(img_bytes)) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = RESIZE_LONG / float(max(w, h))
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        im.save(dest, format="JPEG", quality=JPEG_QUALITY, optimize=True)


def _write_normals(needed_paths: list[Path]) -> int:
    if not needed_paths:
        return 0
    # Prefer the train split (pure-normal) for reference fixtures.
    train_shard = _fetch_shard(PCB1_TRAIN)
    df = _iter_rows(train_shard)
    normals = df[df["label"] == 0] if (df["label"] == 0).any() else df
    if len(normals) < len(needed_paths):
        # fall back to padding from test-normal if the train shard was too small
        test_shard = _fetch_shard(PCB1_TEST)
        test_df = _iter_rows(test_shard)
        normals = (
            [row for _, row in normals.iterrows()]
            + [row for _, row in test_df[test_df["label"] == 0].iterrows()]
        )
        rows_iter = iter(normals)
    else:
        rows_iter = (row for _, row in normals.head(len(needed_paths) * 2).iterrows())

    written = 0
    for dest in needed_paths:
        row = next(rows_iter)
        _save_resized(_extract_bytes(row["image"]), dest)
        written += 1
    return written


def _write_defects(needed_paths: list[Path]) -> int:
    if not needed_paths:
        return 0
    test_shard = _fetch_shard(PCB1_TEST)
    df = _iter_rows(test_shard)
    defects = df[df["label"] == 1]
    if len(defects) < len(needed_paths):
        raise RuntimeError(
            f"VisA pcb1 test split has only {len(defects)} defect rows, "
            f"need {len(needed_paths)}."
        )

    # Pick evenly-spaced indices so we sample across the defect population
    # (VisA does not ship named defect subtypes inside this parquet, but
    # spreading our picks avoids grabbing the same pose repeatedly).
    idx = [int(round(i * (len(defects) - 1) / max(1, len(needed_paths) - 1)))
           for i in range(len(needed_paths))]
    rows = defects.iloc[idx]

    written = 0
    for dest, (_, row) in zip(needed_paths, rows.iterrows()):
        _save_resized(_extract_bytes(row["image"]), dest)
        written += 1
    return written


def main() -> int:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    DEF_DIR.mkdir(parents=True, exist_ok=True)

    ref_paths, def_paths = _expected_paths()
    missing_refs = [p for p in ref_paths if not p.is_file()]
    missing_defs = [p for p in def_paths if not p.is_file()]

    if not missing_refs and not missing_defs:
        print(f"All {len(ref_paths) + len(def_paths)} fixtures already present.")
        return 0

    print(f"Missing references: {len(missing_refs)} | missing defects: {len(missing_defs)}")
    print(f"Source dataset: VisA (CC BY-SA 4.0) via hf://{REPO_ID}")

    n_ref = _write_normals(missing_refs)
    n_def = _write_defects(missing_defs)
    print(f"Wrote {n_ref} reference + {n_def} defect JPEGs under {PCB_ROOT}")

    if not _all_present():
        raise SystemExit("Post-download check failed: not all expected fixtures exist.")

    # Quick sanity check — open every file with PIL and report dims.
    from PIL import Image
    total_bytes = 0
    for p in ref_paths + def_paths:
        with Image.open(p) as im:
            im.verify()
        total_bytes += p.stat().st_size
    print(f"Verified {len(ref_paths) + len(def_paths)} JPEGs, total {total_bytes/1024:.1f} KiB.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
