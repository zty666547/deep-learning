"""Package an exact committed source snapshot; exclude data and pre-existing ZIPs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from microc_foundation.code_bundle import create_code_bundle, source_files_from_tar


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{args.revision}^{{commit}}"], cwd=root, text=True).strip()
    # Read immutable Git objects, never working files or untracked local data.
    payload = subprocess.check_output(["git", "archive", "--format=tar", revision, "--",
                                       "README.md", "pyproject.toml", "requirements.txt", ".gitignore",
                                       "src", "scripts", "tests", "docs", "data/raw/.gitkeep",
                                       "data/processed/.gitkeep", "outputs/.gitkeep",
                                       "deliverables/实验报告.md", "deliverables/figures"], cwd=root)
    receipt = create_code_bundle(source_files_from_tar(payload), Path(args.output), revision)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
