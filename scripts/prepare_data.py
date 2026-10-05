"""Verify or import the benchmark groups used by the default configurations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "catalog.json"


def load_catalog() -> list[dict]:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))["datasets"]


def matching_files(root: Path, patterns: list[str]) -> list[Path]:
    files: set[Path] = set()
    for pattern in patterns:
        files.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def tree_digest(root: Path, patterns: list[str]) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    files = matching_files(root, patterns)
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        total_bytes += len(content)
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return len(files), total_bytes, digest.hexdigest()


def verify(entry: dict, destination: Path) -> tuple[bool, str]:
    count, byte_count, digest = tree_digest(destination, entry["patterns"])
    expected = (
        int(entry["file_count"]),
        int(entry["byte_count"]),
        str(entry["tree_sha256"]),
    )
    actual = (count, byte_count, digest)
    if actual == expected:
        return True, f"{entry['problem']}/{entry['group']}: verified {count} files"
    return False, (
        f"{entry['problem']}/{entry['group']}: verification failed; "
        f"expected count/bytes/hash={expected}, got {actual}"
    )


def copy_selected(entry: dict, source: Path, destination: Path, force: bool) -> None:
    files = matching_files(source, entry["patterns"])
    if not files:
        raise FileNotFoundError(f"No files matching {entry['patterns']} under {source}")
    for source_path in files:
        relative = source_path.relative_to(source)
        target = destination / relative
        if target.exists() and target.read_bytes() != source_path.read_bytes() and not force:
            raise FileExistsError(
                f"Refusing to replace {target}; pass --force after checking the source"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        shutil.copy2(source_path, temporary)
        os.replace(temporary, target)


def select_entries(args, catalog: list[dict]) -> list[dict]:
    if args.all_defaults:
        return catalog
    selected = [
        entry
        for entry in catalog
        if entry["problem"] == args.problem and (args.group is None or entry["group"] == args.group)
    ]
    if not selected:
        raise ValueError(f"No catalog entry for problem={args.problem!r}, group={args.group!r}")
    return selected


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Import manually obtained benchmark files into data/<problem> and "
            "verify the exact default corpus. This command never downloads data."
        )
    )
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--problem", choices=[entry["problem"] for entry in load_catalog()])
    choice.add_argument("--all-defaults", action="store_true")
    parser.add_argument("--group")
    parser.add_argument("--source", type=Path)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="With --all-defaults, read each source from <source-root>/<problem>.",
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    entries = select_entries(args, load_catalog())
    failures = 0
    for entry in entries:
        destination = args.data_root.resolve() / entry["problem"]
        source = args.source
        if args.all_defaults and args.source_root is not None:
            source = args.source_root / entry["problem"]
        if not args.verify_only and source is not None:
            copy_selected(entry, source.resolve(), destination, args.force)
        ok, message = verify(entry, destination)
        print("[PASS]" if ok else "[FAIL]", message)
        if not ok:
            failures += 1
            if source is None and not args.verify_only:
                print(f"       Source: {entry['source_page']}")
                print(f"       Terms:  {entry['redistribution']}")
                print("       Re-run with --source pointing to the expected directory layout.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
