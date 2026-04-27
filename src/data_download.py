"""
Step 1: Download the JARVIS dataset used by the project.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import ssl
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from utils import DATA_DIR, RAW_DATA_DIR, RESULTS_DIR, ensure_directories, write_json

JARVIS_URL = "https://ndownloader.figshare.com/files/27732228"
ZIP_PATH = DATA_DIR / "jarvis_alignn.zip"
MANIFEST_PATH = RESULTS_DIR / "download_manifest.json"


def compute_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _report_progress(downloaded: int, total: int) -> None:
    if total > 0:
        percent = min(downloaded / total * 100.0, 100.0)
        print(
            f"\r📥 Downloading JARVIS dataset: {percent:5.1f}% "
            f"({downloaded / 1024**2:.1f} / {total / 1024**2:.1f} MB)",
            end="",
        )
    else:
        print(f"\r📥 Downloading JARVIS dataset: {downloaded / 1024**2:.1f} MB", end="")


def download_file(url: str, destination: Path, timeout: int = 60) -> Path:
    ensure_directories()
    destination.parent.mkdir(parents=True, exist_ok=True)

    context = ssl.create_default_context()
    with urllib.request.urlopen(url, timeout=timeout, context=context) as response:
        total = int(response.headers.get("Content-Length", "0") or 0)
        downloaded = 0
        with destination.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                _report_progress(downloaded, total)
    print()
    return destination


def verify_zip_file(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Downloaded archive not found: {path}")

    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise zipfile.BadZipFile(f"Corrupt member found in archive: {bad_member}")

    checksum = compute_sha256(path)
    file_size = path.stat().st_size
    manifest = {
        "archive": str(path),
        "sha256": checksum,
        "size_bytes": file_size,
        "verified": True,
    }
    write_json(MANIFEST_PATH, manifest)
    print(f"🔐 SHA256 checksum: {checksum}")
    return manifest


def extract_archive(path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(destination)

    if RAW_DATA_DIR.exists():
        return RAW_DATA_DIR

    candidates = sorted(
        subdir
        for subdir in destination.iterdir()
        if subdir.is_dir() and (subdir / "id_prop.csv").exists()
    )
    if not candidates:
        raise FileNotFoundError("Could not locate extracted JARVIS dataset folder.")

    extracted_dir = candidates[0]
    if extracted_dir != RAW_DATA_DIR:
        if RAW_DATA_DIR.exists():
            shutil.rmtree(RAW_DATA_DIR)
        extracted_dir.rename(RAW_DATA_DIR)
    return RAW_DATA_DIR


def download_jarvis_dataset(force: bool = False) -> Path:
    ensure_directories()

    if (RAW_DATA_DIR / "id_prop.csv").exists() and not force:
        print(f"✅ Dataset already available at {RAW_DATA_DIR}")
        return RAW_DATA_DIR

    if force:
        for target in (ZIP_PATH, RAW_DATA_DIR):
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()

    try:
        return download_via_jarvis_api()
    except Exception as exc:
        print(f"⚠️ jarvis-tools API download path failed: {exc}")
        print("🔁 Falling back to the legacy archive URL.")

    try:
        print(f"🌐 Source: {JARVIS_URL}")
        archive = download_file(JARVIS_URL, ZIP_PATH)
        verify_zip_file(archive)
        extracted = extract_archive(archive, DATA_DIR)
        print(f"📦 Extracted dataset to {extracted}")
        print(f"📄 Metadata file: {extracted / 'id_prop.csv'}")
        return extracted
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Network download failed: {exc}") from exc
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"Downloaded archive is invalid: {exc}") from exc


def _material_record_from_jarvis(row: dict[str, object]) -> dict[str, object]:
    atoms = row.get("atoms") or {}
    elements = atoms.get("elements") or []
    lattice = atoms.get("lattice_mat") or atoms.get("lattice") or []
    volume = row.get("volume")
    if volume is None and len(lattice) == 3:
        try:
            import numpy as np

            volume = float(abs(np.linalg.det(np.asarray(lattice, dtype=float))))
        except Exception:
            volume = None

    return {
        "jid": row.get("jid", ""),
        "formula": row.get("formula", ""),
        "nsites": len(elements),
        "spacegroup": row.get("spg_symbol") or row.get("spacegroup_symbol") or "",
        "optb88vdw_bandgap": row.get("optb88vdw_bandgap"),
        "formation_energy_peratom": row.get("formation_energy_peratom"),
        "volume": volume,
        "dimensionality": row.get("dimensionality"),
        "mbj_bandgap": row.get("mbj_bandgap"),
        "hse_gap": row.get("hse_gap"),
        "exfoliation_energy": row.get("exfoliation_energy"),
    }


def download_via_jarvis_api() -> Path:
    try:
        from jarvis.db.figshare import data
    except Exception as exc:
        raise RuntimeError(f"jarvis-tools is not installed: {exc}") from exc

    print("🌐 Source: jarvis.db.figshare.data('dft_3d')")
    rows = data("dft_3d")
    if not rows:
        raise RuntimeError("jarvis-tools returned an empty dataset.")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    for index, row in enumerate(rows, start=1):
        jid = row.get("jid")
        atoms = row.get("atoms")
        if not jid or not atoms:
            continue

        (RAW_DATA_DIR / f"{jid}.json").write_text(json.dumps(atoms))
        csv_rows.append(_material_record_from_jarvis(row))
        if index % 5000 == 0:
            print(f"🧱 Materialized {index:,} / {len(rows):,} structures")

    fieldnames = list(csv_rows[0].keys())
    with (RAW_DATA_DIR / "id_prop.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    manifest = {
        "source": "jarvis.db.figshare.data('dft_3d')",
        "records": len(csv_rows),
        "raw_dir": str(RAW_DATA_DIR),
        "verified": True,
    }
    write_json(MANIFEST_PATH, manifest)
    print(f"📦 Materialized {len(csv_rows):,} structures into {RAW_DATA_DIR}")
    return RAW_DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the JARVIS dataset")
    parser.add_argument("--force", action="store_true", help="Re-download the dataset")
    args = parser.parse_args()

    try:
        download_jarvis_dataset(force=args.force)
    except Exception as exc:
        print(f"❌ Download step failed: {exc}")
        raise


if __name__ == "__main__":
    main()
