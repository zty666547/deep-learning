"""Verifiable source-only course ZIPs from immutable Git archive members."""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

PREFIX = "deep-learning/"
ROOT_FILES = {"README.md", "pyproject.toml", "requirements.txt", ".gitignore"}
REQUIRED = {"README.md", "pyproject.toml", "requirements.txt",
            "scripts/minimal_pipeline.py", "src/microc_foundation/__init__.py"}
EMPTY_DATA = {"data/raw/.gitkeep", "data/processed/.gitkeep", "outputs/.gitkeep"}


def include_member(name: str) -> bool:
    path = PurePosixPath(name)
    if (path.is_absolute() or ".." in path.parts or str(path) != name or not path.parts
            or "\\" in name or "\x00" in name):
        raise ValueError(f"unsafe archive member: {name}")
    if name in ROOT_FILES or name in EMPTY_DATA:
        return True
    if name == "deliverables/实验报告.md":
        return True
    if path.parent == PurePosixPath("deliverables/figures") and path.suffix == ".png":
        return True
    if any(part.startswith(".") or part == "__pycache__" for part in path.parts):
        return False
    return path.parts[0] in {"src", "scripts", "tests", "docs"}


def source_files_from_tar(payload: bytes) -> dict[str, bytes]:
    files = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        for member in archive:
            name = member.name.rstrip("/") if member.isdir() else member.name
            selected = include_member(name)
            if not selected or member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"non-regular source archive member: {member.name}")
            if member.name in files:
                raise ValueError(f"duplicate archive member: {member.name}")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"unreadable member: {member.name}")
            files[member.name] = handle.read()
    return files


def create_code_bundle(files: dict[str, bytes], output: Path, revision: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("source revision must be an exact Git commit SHA")
    selected = {name: payload for name, payload in files.items() if include_member(name)}
    if not REQUIRED.issubset(selected):
        raise ValueError(f"missing required source files: {sorted(REQUIRED - set(selected))}")
    manifest = {
        "schema": "microc-source-bundle-v1", "source_commit": revision,
        "raw_data_included": False, "trained_weights_included": False,
        "data_setup": "Place real inputs in data/raw as documented in README; do not use test fixtures as results.",
        "files": {name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                  for name, payload in sorted(selected.items())},
    }
    members = {**selected, "bundle-manifest.json": json.dumps(
        manifest, indent=2, ensure_ascii=False, sort_keys=True).encode("utf-8")}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted(members.items()):
            item = zipfile.ZipInfo(PREFIX + name, date_time=(1980, 1, 1, 0, 0, 0))
            item.create_system = 3
            item.external_attr = 0o100644 << 16
            item.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(item, payload)
    return verify_code_bundle(output)


def verify_code_bundle(path: Path) -> dict:
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise ValueError("duplicate or damaged ZIP member")
        manifest = json.loads(archive.read(PREFIX + "bundle-manifest.json"))
        if manifest.get("schema") != "microc-source-bundle-v1":
            raise ValueError("unknown bundle schema")
        if (not re.fullmatch(r"[0-9a-f]{40}", manifest.get("source_commit", ""))
                or manifest.get("raw_data_included") is not False
                or manifest.get("trained_weights_included") is not False):
            raise ValueError("invalid source revision or data inclusion policy")
        expected = {PREFIX + name for name in manifest["files"]} | {PREFIX + "bundle-manifest.json"}
        if set(names) != expected:
            raise ValueError("ZIP contents differ from manifest")
        if not REQUIRED.issubset(manifest["files"]):
            raise ValueError("bundle lacks required source files")
        for name, record in manifest["files"].items():
            if not include_member(name):
                raise ValueError(f"excluded member present in ZIP: {name}")
            payload = archive.read(PREFIX + name)
            if len(payload) != record["bytes"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
                raise ValueError(f"bundle integrity mismatch: {name}")
    return {"status": "source_bundle_integrity_passed", "source_commit": manifest["source_commit"],
            "source_files": len(manifest["files"]), "zip_members": len(names),
            "archive_bytes": path.stat().st_size,
            "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "raw_data_included": False, "trained_weights_included": False,
            "full_experiment_rerun": False}
