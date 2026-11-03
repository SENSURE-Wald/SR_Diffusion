#!/usr/bin/env python3
"""
Rename files whose names look like query strings, e.g.:

    massen.php?file=dop20rgbi_32_550_6059_1_sh_2025.tif&id=7&live=2025&km=32550_6050&stack=32550_6059

to the value of their 'file=' parameter, e.g.:

    dop20rgbi_32_550_6059_1_sh_2025.tif

Usage:
    python rename_by_file_param.py /path/to/directory
    python rename_by_file_param.py /path/to/directory --dry-run
    python rename_by_file_param.py /path/to/directory --recursive
"""

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote

# Matches "file=" followed by everything up to the next "&" or end of string
FILE_PARAM_RE = re.compile(r"file=([^&]+)")


def extract_target_name(filename: str) -> str | None:
    """Extract the value of the file= parameter from a filename, or None if not found."""
    match = FILE_PARAM_RE.search(filename)
    if not match:
        return None
    return unquote(match.group(1))


def unique_path(path: Path) -> Path:
    """If path already exists, append _1, _2, ... before the extension to avoid clobbering."""
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    counter = 1
    while True:
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def rename_files(directory: Path, recursive: bool, dry_run: bool) -> None:
    pattern = "**/*" if recursive else "*"
    files = [p for p in directory.glob(pattern) if p.is_file()]

    if not files:
        print(f"No files found in {directory}")
        return

    renamed, skipped = 0, 0

    for path in files:
        target_name = extract_target_name(path.name)

        if not target_name:
            print(f"SKIP (no file= param): {path.name}")
            skipped += 1
            continue

        new_path = path.with_name(target_name)

        if new_path == path:
            print(f"SKIP (already correct): {path.name}")
            skipped += 1
            continue

        new_path = unique_path(new_path)

        if dry_run:
            print(f"[DRY RUN] {path.name}  ->  {new_path.name}")
        else:
            path.rename(new_path)
            print(f"RENAMED: {path.name}  ->  {new_path.name}")
        renamed += 1

    print(f"\nDone. {renamed} file(s) renamed, {skipped} skipped.")


def main():
    parser = argparse.ArgumentParser(description="Rename files based on their embedded file= parameter.")
    parser.add_argument("directory", type=str, help="Directory containing files to rename")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be renamed without doing it")
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"Error: {directory} is not a valid directory", file=sys.stderr)
        sys.exit(1)

    rename_files(directory, args.recursive, args.dry_run)


if __name__ == "__main__":
    main()
