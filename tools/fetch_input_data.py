#!/usr/bin/env python3
"""Download the HPC-Bench input data from Hugging Face and put it in place.

Every archive in the Hugging Face dataset holds one benchmark's ``input_data/``
directory. This script downloads the archives you ask for, checks each one
against the SHA-256 in ``manifest.csv``, and extracts it into
``EX<N>/<benchmark>/input_data/``. Archives are deleted after extraction unless
``--keep-archives`` is given.

Examples:
    python tools/fetch_input_data.py                          # everything (~38 GB zipped)
    python tools/fetch_input_data.py --ex EX1                 # one setting
    python tools/fetch_input_data.py --ex EX1,EX2 --benchmarks 2mm,bfs
    python tools/fetch_input_data.py --list                   # show what is available
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    sys.exit("huggingface_hub is required: pip install huggingface_hub")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "bcui2/HPC-Bench"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ex", default="EX1,EX2,EX3", help="Comma-separated settings (default: all).")
    p.add_argument("--benchmarks", default="all", help="Comma-separated benchmark names (default: all).")
    p.add_argument("--repo", default=DEFAULT_REPO, help=f"Hugging Face dataset (default: {DEFAULT_REPO}).")
    p.add_argument("--revision", default=None, help="Dataset revision (branch, tag, or commit).")
    p.add_argument("--download-dir", default=str(REPO_ROOT / ".hf_input_data"),
                   help="Where archives are downloaded before extraction.")
    p.add_argument("--workers", type=int, default=4, help="Parallel downloads (default: 4).")
    p.add_argument("--force", action="store_true", help="Re-extract even if input_data/ already exists.")
    p.add_argument("--keep-archives", action="store_true", help="Keep the downloaded zip files.")
    p.add_argument("--list", action="store_true", help="List matching archives and exit.")
    return p.parse_args()


def load_manifest(args: argparse.Namespace) -> list[dict]:
    path = hf_hub_download(args.repo, "manifest.csv", repo_type="dataset", revision=args.revision,
                           local_dir=args.download_dir)
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def select(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    exs = {e.strip() for e in args.ex.split(",") if e.strip()}
    rows = [r for r in rows if r["ex"] in exs]
    if args.benchmarks != "all":
        wanted = {b.strip() for b in args.benchmarks.split(",") if b.strip()}
        unknown = wanted - {r["benchmark"] for r in rows}
        if unknown:
            sys.exit(f"No input data for {sorted(unknown)} in {sorted(exs)}. Use --list to see what is available.")
        rows = [r for r in rows if r["benchmark"] in wanted]
    return rows


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_one(row: dict, args: argparse.Namespace) -> str:
    target = REPO_ROOT / row["ex"] / row["benchmark"]
    label = f"{row['ex']}/{row['benchmark']}"
    if not target.is_dir():
        return f"skip   {label}: benchmark directory not found"
    if (target / "input_data").exists() and not args.force:
        return f"exists {label}"

    archive = Path(hf_hub_download(args.repo, row["path"], repo_type="dataset", revision=args.revision,
                                   local_dir=args.download_dir))
    if sha256(archive) != row["sha256"]:
        archive.unlink()
        raise RuntimeError(f"{label}: checksum mismatch, archive removed; re-run to download it again")

    if (target / "input_data").exists():
        shutil.rmtree(target / "input_data")
    with zipfile.ZipFile(archive) as z:
        z.extractall(target)
    if not args.keep_archives:
        archive.unlink()
    return f"done   {label}"


def main() -> None:
    args = parse_args()
    rows = select(load_manifest(args), args)
    total_gb = sum(int(r["bytes"]) for r in rows) / 1e9

    if args.list:
        for r in rows:
            print(f"{r['ex']}/{r['benchmark']:<24} {int(r['bytes']) / 1e6:10.1f} MB")
        print(f"{len(rows)} archives, {total_gb:.1f} GB zipped")
        return

    print(f"Fetching {len(rows)} archives ({total_gb:.1f} GB zipped) from {args.repo}")
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, r, args): r for r in rows}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                print(f"[{i}/{len(rows)}] {fut.result()}", flush=True)
            except Exception as exc:  # report and keep going
                failed.append(str(exc))
                print(f"[{i}/{len(rows)}] FAILED {exc}", flush=True)

    if failed:
        sys.exit(f"{len(failed)} archive(s) failed; re-run the same command to retry them.")
    print("Input data is ready.")


if __name__ == "__main__":
    main()
