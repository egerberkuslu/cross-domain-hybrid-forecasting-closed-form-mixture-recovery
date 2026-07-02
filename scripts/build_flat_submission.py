"""Build Elsevier-EM-compliant flat-structure source bundle for a paper.

Elsevier Editorial Manager rejects LaTeX submissions that contain
subfolders. This script:

  1. Copies main.tex, refs.bib, highlights.tex, cover_letter.tex into
     a temporary flat directory.
  2. Copies every file from sections/, tables/, figures/ into the same
     flat directory (no subdirs).
  3. Rewrites \\input{sections/x.tex} -> \\input{x.tex},
     \\input{tables/x.tex} -> \\input{x.tex},
     \\includegraphics[...]{figures/x.pdf} -> \\includegraphics[...]{x.pdf}.
  4. Compiles once to confirm the flat tree still builds.
  5. Wraps the flat dir as source.tar.gz, ready to upload to EM.

Usage:
  python scripts/build_flat_submission.py paper_a
  python scripts/build_flat_submission.py paper_b
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def flatten(src_paper: Path, dst_flat: Path) -> None:
    dst_flat.mkdir(parents=True, exist_ok=True)
    for item in ["main.tex", "refs.bib", "highlights.tex", "cover_letter.tex"]:
        p = src_paper / item
        if p.exists():
            shutil.copy2(p, dst_flat / p.name)
    for sub in ["sections", "tables", "figures"]:
        d = src_paper / sub
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.is_file():
                target = dst_flat / f.name
                if target.exists():
                    print(f"  WARN: name collision on {f.name}, renaming")
                    target = dst_flat / f"{sub}_{f.name}"
                shutil.copy2(f, target)


def rewrite_paths(flat_dir: Path) -> None:
    pat_input = re.compile(r"\\input\{(sections|tables)/([^}]+)\}")
    pat_graphics = re.compile(r"(\\includegraphics(?:\[[^\]]*\])?)\{figures/([^}]+)\}")
    for tex in flat_dir.glob("*.tex"):
        src = tex.read_text()
        out = pat_input.sub(r"\\input{\2}", src)
        out = pat_graphics.sub(r"\1{\2}", out)
        if out != src:
            tex.write_text(out)
            print(f"  rewrote paths in {tex.name}")


def compile_check(flat_dir: Path) -> bool:
    log = []
    for cmd in [
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["bibtex", "main"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
    ]:
        try:
            r = subprocess.run(
                cmd,
                cwd=flat_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
            )
            log.append((cmd[0], r.returncode))
            if r.returncode != 0:
                print(f"  COMPILE FAILED at {cmd[0]}; last 30 stdout lines:")
                for line in r.stdout.splitlines()[-30:]:
                    print(f"    {line}")
                return False
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT at {cmd[0]}")
            return False
    print(f"  compile OK: {log}")
    pdf = flat_dir / "main.pdf"
    if pdf.exists():
        size_kb = pdf.stat().st_size // 1024
        print(f"  main.pdf -> {size_kb} KB")
        return True
    return False


def make_tarball(flat_dir: Path, out_tar: Path) -> None:
    # Strip build artefacts before archiving
    for pat in [
        "*.aux",
        "*.log",
        "*.bbl",
        "*.blg",
        "*.out",
        "*.fls",
        "*.fdb_latexmk",
        "*.synctex.gz",
        "*.spl",
    ]:
        for f in flat_dir.glob(pat):
            f.unlink()
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_tar, "w:gz") as tar:
        for f in sorted(flat_dir.iterdir()):
            tar.add(f, arcname=f.name)
    size_kb = out_tar.stat().st_size // 1024
    print(f"  tarball: {out_tar} ({size_kb} KB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paper", help="paper_a or paper_b (dir under repo root)")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument(
        "--keep-flat-dir", action="store_true", help="keep flat dir for inspection"
    )
    args = ap.parse_args()

    repo = Path(args.repo)
    src = repo / args.paper
    if not src.is_dir():
        print(f"missing: {src}")
        return 2

    sub_dir = repo / "submission" / args.paper
    sub_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"flat_{args.paper}_") as tmp:
        flat = Path(tmp) / "flat"
        print(f"[{args.paper}] flattening -> {flat}")
        flatten(src, flat)
        print(f"[{args.paper}] rewriting input/includegraphics paths")
        rewrite_paths(flat)
        print(f"[{args.paper}] compile check")
        if not compile_check(flat):
            print(f"[{args.paper}] FLAT BUILD FAILED")
            return 1
        # Copy the freshly compiled flat PDF over as the canonical manuscript.pdf
        shutil.copy2(flat / "main.pdf", sub_dir / "manuscript.pdf")
        print(f"  manuscript.pdf -> {sub_dir / 'manuscript.pdf'}")
        out_tar = sub_dir / "source.tar.gz"
        print(f"[{args.paper}] packaging tarball")
        make_tarball(flat, out_tar)
        if args.keep_flat_dir:
            keep = repo / "submission" / args.paper / "_flat_preview"
            if keep.exists():
                shutil.rmtree(keep)
            shutil.copytree(flat, keep)
            print(f"  flat dir preview: {keep}")
    print(f"[{args.paper}] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
