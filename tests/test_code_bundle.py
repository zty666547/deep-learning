import io
import json
import tarfile
import zipfile

import pytest

from microc_foundation.code_bundle import (
    PREFIX,
    REQUIRED,
    create_code_bundle,
    include_member,
    source_files_from_tar,
    verify_code_bundle,
)


def files():
    return {name: f"fixture {name}".encode() for name in REQUIRED}


def test_source_bundle_is_deterministic_and_preserves_unicode_and_binary(tmp_path):
    content = {**files(), "docs/中文.md": "真实配图".encode(), "docs/figures/one.png": b"\x00\xff\x01",
               "deliverables/实验报告.md": "正式正文".encode(), "deliverables/figures/one.png": b"png"}
    first = create_code_bundle(content, tmp_path / "first.zip", "a" * 40)
    second = create_code_bundle(content, tmp_path / "second.zip", "a" * 40)
    assert first["archive_sha256"] == second["archive_sha256"]
    assert first["source_files"] == len(content)
    assert not first["raw_data_included"] and not first["full_experiment_rerun"]
    with zipfile.ZipFile(tmp_path / "first.zip") as archive:
        assert archive.read(PREFIX + "docs/中文.md") == content["docs/中文.md"]


def test_excludes_private_data_environment_outputs_and_recursive_deliverables(tmp_path):
    excluded = {".env", ".git/config", "src/.env", "src/__pycache__/a.pyc",
                "data/raw/real.cool", "outputs/model.pt", "deliverables/report.zip"}
    content = {**files(), **{name: b"private" for name in excluded}, "data/raw/.gitkeep": b""}
    receipt = create_code_bundle(content, tmp_path / "safe.zip", "a" * 40)
    assert receipt["source_files"] == len(REQUIRED) + 1
    with zipfile.ZipFile(tmp_path / "safe.zip") as archive:
        assert not any(PREFIX + name in archive.namelist() for name in excluded)


def test_rejects_unsafe_member_paths():
    for name in ["../secret", "/absolute", "src/../secret", "src/./one.py", "",
                 "src\\evil.py", "src/\\..\\evil.py", "src/a\x00b"]:
        with pytest.raises(ValueError):
            include_member(name)


def test_tar_rejects_links_without_extracting_them():
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        item = tarfile.TarInfo("src/link.py")
        item.type = tarfile.SYMTYPE
        item.linkname = "/private/secret"
        archive.addfile(item)
    with pytest.raises(ValueError, match="non-regular"):
        source_files_from_tar(stream.getvalue())


def test_detects_tampered_source_payload(tmp_path):
    original = tmp_path / "original.zip"
    create_code_bundle(files(), original, "a" * 40)
    with zipfile.ZipFile(original) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members[PREFIX + "README.md"] = b"changed"
    with zipfile.ZipFile(tmp_path / "changed.zip", "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    with pytest.raises(ValueError, match="integrity mismatch"):
        verify_code_bundle(tmp_path / "changed.zip")


def test_refuses_to_overwrite_existing_bundle(tmp_path):
    path = tmp_path / "original.zip"
    create_code_bundle(files(), path, "a" * 40)
    saved = path.read_bytes()
    with pytest.raises(FileExistsError):
        create_code_bundle(files(), path, "b" * 40)
    assert path.read_bytes() == saved


def test_rejects_incomplete_source_and_invalid_revision(tmp_path):
    with pytest.raises(ValueError, match="missing required"):
        create_code_bundle({"README.md": b"one"}, tmp_path / "bad.zip", "a" * 40)
    with pytest.raises(ValueError, match="exact Git"):
        create_code_bundle(files(), tmp_path / "bad.zip", "main")
    assert not (tmp_path / "bad.zip").exists()
    valid = tmp_path / "valid.zip"
    create_code_bundle(files(), valid, "a" * 40)
    with zipfile.ZipFile(valid) as archive:
        manifest = json.loads(archive.read(PREFIX + "bundle-manifest.json"))
    assert manifest["source_commit"] == "a" * 40
