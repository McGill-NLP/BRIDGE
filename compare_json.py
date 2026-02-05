#!/usr/bin/env python3
"""Compare two JSON or JSONL files for semantic equality."""

import argparse
import json
import sys
import difflib
from itertools import zip_longest
from pathlib import Path
from typing import Literal, Tuple, Union

JsonType = Union[dict, list, int, float, str, bool, None]
Kind = Literal["json", "jsonl"]


def load_json_or_jsonl(path: Path) -> Tuple[Kind, Union[JsonType, list[JsonType]]]:
    """Load a JSON or JSONL file."""
    if path.suffix.lower() == ".jsonl":
        data: list[JsonType] = []
        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    print(f"Error parsing {path}:{lineno}: {exc}", file=sys.stderr)
                    sys.exit(2)
        return "jsonl", data
    else:
        try:
            with path.open(encoding="utf-8") as f:
                return "json", json.load(f)
        except json.JSONDecodeError as exc:
            print(f"Error parsing {path}: {exc}", file=sys.stderr)
            sys.exit(2)


def normalize(value: JsonType) -> JsonType:
    """Normalize nested dict order by sorting keys recursively."""
    if isinstance(value, dict):
        return {k: normalize(value[k]) for k in sorted(value.keys())}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    return value


def truncate_line(text: str, limit: int = 50) -> str:
    """Return the line limited to the first `limit` characters."""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def print_json_diff(obj1: JsonType, obj2: JsonType, path1: Path, path2: Path) -> None:
    """Print a unified diff of two JSON objects."""
    pretty1 = json.dumps(normalize(obj1), sort_keys=True, indent=2, ensure_ascii=False)
    pretty2 = json.dumps(normalize(obj2), sort_keys=True, indent=2, ensure_ascii=False)
    for line in unified_diff_lines(pretty1.splitlines(), pretty2.splitlines(), path1, path2):
        print(line)


def print_jsonl_diff(list1: list[JsonType], list2: list[JsonType], path1: Path, path2: Path) -> None:
    """Print differing lines between two JSONL files."""
    norm1 = [json.dumps(normalize(obj), sort_keys=True, ensure_ascii=False) for obj in list1]
    norm2 = [json.dumps(normalize(obj), sort_keys=True, ensure_ascii=False) for obj in list2]
    matcher = difflib.SequenceMatcher(None, norm1, norm2)
    has_diff = False

    def show_pair(line1: str | None, line2: str | None, idx1: int | None, idx2: int | None) -> None:
        nonlocal has_diff
        if line1 is not None:
            print(f"- {path1}:{idx1}: {truncate_line(line1)}")
        if line2 is not None:
            print(f"+ {path2}:{idx2}: {truncate_line(line2)}")
        if line1 is not None and line2 is not None:
            for diff_line in difflib.ndiff([line1], [line2]):
                print(f"    {diff_line.rstrip()}")
        elif line1 is not None:
            print("    (deleted)")
        else:
            print("    (inserted)")
        has_diff = True

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            for offset, (l1, l2) in enumerate(zip_longest(norm1[i1:i2], norm2[j1:j2])):
                idx1 = i1 + offset + 1 if l1 is not None else None
                idx2 = j1 + offset + 1 if l2 is not None else None
                show_pair(l1, l2, idx1, idx2)
        elif tag == "delete":
            for offset, line in enumerate(norm1[i1:i2]):
                show_pair(line, None, i1 + offset + 1, None)
        elif tag == "insert":
            for offset, line in enumerate(norm2[j1:j2]):
                show_pair(None, line, None, j1 + offset + 1)

    if not has_diff:
        print("(No textual differences; files may contain binary data)")


def unified_diff_lines(lines1: list[str], lines2: list[str], path1: Path, path2: Path) -> list[str]:
    return list(
        difflib.unified_diff(lines1, lines2, fromfile=str(path1), tofile=str(path2), lineterm="")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two JSON/JSONL files for equality (ignoring key order).")
    parser.add_argument("file1", type=Path, help="First JSON/JSONL file to compare")
    parser.add_argument("file2", type=Path, help="Second JSON/JSONL file to compare")
    args = parser.parse_args()

    if not args.file1.exists():
        print(f"Error: {args.file1} does not exist", file=sys.stderr)
        sys.exit(2)
    if not args.file2.exists():
        print(f"Error: {args.file2} does not exist", file=sys.stderr)
        sys.exit(2)

    kind1, data1 = load_json_or_jsonl(args.file1)
    kind2, data2 = load_json_or_jsonl(args.file2)

    if kind1 != kind2:
        print("Files differ (one is JSON, the other is JSONL)")
        sys.exit(1)

    if kind1 == "json":
        if normalize(data1) == normalize(data2):
            print("Files are equivalent")
            sys.exit(0)
        print("Files differ")
        print_json_diff(data1, data2, args.file1, args.file2)
    else:
        if [normalize(obj) for obj in data1] == [normalize(obj) for obj in data2]:
            print("Files are equivalent")
            sys.exit(0)
        print_jsonl_diff(data1, data2, args.file1, args.file2)

    sys.exit(1)


if __name__ == "__main__":
    main()
