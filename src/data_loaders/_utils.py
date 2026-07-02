"""Shared helpers for dataset loaders: streaming download + tar extraction."""
from __future__ import annotations

import hashlib
import logging
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable

import requests
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# downloading
# ----------------------------------------------------------------------


def stream_download(
    url: str,
    dest: str | os.PathLike,
    chunk_size: int = 1 << 20,
    expected_size: int | None = None,
    skip_if_exists: bool = True,
    timeout: int = 60,
) -> Path:
    """Download `url` to `dest` with a tqdm progress bar.

    If the destination file already exists and has the expected size (or the
    expected size is unknown) we skip the download. Returns the destination
    path.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if skip_if_exists and dest.exists():
        if expected_size is None or dest.stat().st_size == expected_size:
            logger.info("Found cached download: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
            return dest
        logger.warning(
            "Existing file %s has size %s but expected %s — re-downloading.",
            dest, dest.stat().st_size, expected_size,
        )

    logger.info("Downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or expected_size or 0)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as f, tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            desc=dest.name,
            mininterval=2.0,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        tmp.replace(dest)

    logger.info("Saved %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


def md5_of(path: str | os.PathLike, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------
# extraction
# ----------------------------------------------------------------------


def safe_extract_tar(
    archive: str | os.PathLike,
    dest_dir: str | os.PathLike,
    members: Iterable[str] | None = None,
) -> Path:
    """Extract a tar(.gz / .bz2) archive safely (rejects path traversal)."""
    archive = Path(archive)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    resolved_root = dest_dir.resolve()

    logger.info("Extracting %s -> %s", archive, dest_dir)
    with tarfile.open(archive, "r:*") as tar:
        wanted = set(members) if members is not None else None
        for m in tar.getmembers():
            if wanted is not None and m.name not in wanted:
                continue
            target = (dest_dir / m.name).resolve()
            if not str(target).startswith(str(resolved_root)):
                raise RuntimeError(f"Refusing to extract outside dest: {m.name}")
        tar.extractall(dest_dir)
    return dest_dir


def safe_extract_zip(archive: str | os.PathLike, dest_dir: str | os.PathLike) -> Path:
    archive = Path(archive)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    resolved_root = dest_dir.resolve()
    logger.info("Extracting %s -> %s", archive, dest_dir)
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            target = (dest_dir / name).resolve()
            if not str(target).startswith(str(resolved_root)):
                raise RuntimeError(f"Refusing to extract outside dest: {name}")
        zf.extractall(dest_dir)
    return dest_dir
