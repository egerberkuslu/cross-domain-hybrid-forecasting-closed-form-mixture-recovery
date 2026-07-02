"""GEANT 15-min traffic-matrix loader (TOTEM dataset).

Each XML file ``traffic-matrices/IntraTM-YYYY-MM-DD-HH-MM.xml`` is a single
15-minute snapshot containing a full OD matrix in kbps. We sum *all* dst
values across all src blocks to obtain the network's aggregate total
traffic for that 15-minute bin (units stay in kbps; the magnitude is
preserved across the resample-and-sum to hourly).

Source: Uhlig et al., ACM CCR 2006. Hosted on totem.run.montefiore.ulg.ac.be.
"""
from __future__ import annotations

import logging
import re
import tarfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseLoader, DatasetSpec
from ._utils import stream_download

logger = logging.getLogger(__name__)


_URL = "http://totem.run.montefiore.ulg.ac.be/files/data/traffic-matrices-anonymized-v2.tar.bz2"
_ARCHIVE_NAME = "traffic-matrices-anonymized-v2.tar.bz2"
_FILENAME_RE = re.compile(r"IntraTM-(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\.xml$")


class GeantLoader(BaseLoader):
    def __init__(self, raw_dir: str | Path, processed_path: str | Path, resample_to: str = "1h"):
        super().__init__(
            DatasetSpec(
                name="geant",
                raw_dir=Path(raw_dir),
                processed_path=Path(processed_path),
                native_freq="15min",
                resample_to=resample_to,
                description="GEANT 15-min OD kbps summed across all src/dst pairs (TOTEM v2).",
            )
        )

    def download(self) -> None:
        dest = self.spec.raw_dir / _ARCHIVE_NAME
        if not dest.exists():
            stream_download(_URL, dest)
        else:
            logger.info("[geant] cached %s (%.1f MB)", dest, dest.stat().st_size / 1e6)

    @staticmethod
    def _parse_one_xml(xml_bytes: bytes) -> tuple[datetime | None, float | None]:
        """Return (timestamp, total_kbps) for a single IntraTM XML."""
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as e:
            logger.warning("[geant] XML parse error: %s", e)
            return None, None

        # date — first try `info/date`, else fall back to filename
        date_text = None
        info = root.find("info")
        if info is not None:
            d = info.find("date")
            if d is not None:
                date_text = d.text
        if date_text:
            try:
                ts = datetime.fromisoformat(date_text.strip())
            except ValueError:
                ts = None
        else:
            ts = None

        # sum every <dst>...</dst> text inside any IntraTM/src/dst
        total = 0.0
        any_value = False
        for src in root.iter("src"):
            for dst in src.findall("dst"):
                try:
                    total += float(dst.text)
                    any_value = True
                except (TypeError, ValueError):
                    continue
        return ts, total if any_value else None

    def parse(self) -> pd.DataFrame:
        archive = self.spec.raw_dir / _ARCHIVE_NAME
        rows = []
        with tarfile.open(archive, "r:bz2") as tar:
            members = [
                m for m in tar.getmembers()
                if m.name.startswith("traffic-matrices/IntraTM-") and m.name.endswith(".xml")
            ]
            logger.info("[geant] %d IntraTM XML files in archive", len(members))
            for i, m in enumerate(members, 1):
                xml_bytes = tar.extractfile(m).read()
                ts, val = self._parse_one_xml(xml_bytes)
                # fall back to filename-derived timestamp if needed
                if ts is None:
                    match = _FILENAME_RE.search(m.name)
                    if match:
                        y, mo, d, h, mi = map(int, match.groups())
                        ts = datetime(y, mo, d, h, mi)
                if ts is not None and val is not None:
                    rows.append((ts, val))
                if i % 2000 == 0:
                    logger.info("[geant] parsed %d/%d files", i, len(members))
            logger.info("[geant] parsed all %d files", len(members))

        df = pd.DataFrame(rows, columns=["timestamp", "value"]).set_index("timestamp")
        df = df.sort_index()
        if df.index.has_duplicates:
            df = df.groupby(level=0).mean()
        df["value"] = df["value"].astype("float64")
        logger.info("[geant] series: %d rows, %s -> %s, mean=%.3e",
                    len(df), df.index.min(), df.index.max(), float(df["value"].mean()))
        return df
