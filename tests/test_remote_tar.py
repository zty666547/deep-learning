import io
import runpy
import tarfile
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pytest

module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/inspect_remote_tar.py"))
inspect_tar = module["inspect_tar"]
read_range = module["read_range"]


def _archive(*, long_name=False):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.GNU_FORMAT) as archive:
        for name in ("a.cool.gz", "x" * 120 + ".cool.gz" if long_name else "b.cool.gz"):
            info = tarfile.TarInfo(name)
            info.size = 2048
            archive.addfile(info, io.BytesIO(b"X" * info.size))
    return output.getvalue()


def test_tar_inspector_skips_file_payloads():
    data = _archive()
    requests = []

    def fetch(url, start, size):
        requests.append((start, size))
        return data[start:start + size], len(data)

    with patch.dict(inspect_tar.__globals__, read_range=fetch):
        result = inspect_tar("https://example.org/data.tar")
    assert [row["name"] for row in result["files"]] == ["a.cool.gz", "b.cool.gz"]
    assert requests == [(0, 512), (2560, 512), (5120, 512)]
    assert result["metadata_bytes_read"] == 1536
    assert result["complete"] is True


def test_tar_inspector_supports_gnu_long_names():
    data = _archive(long_name=True)
    with patch.dict(inspect_tar.__globals__, read_range=lambda url, start, size: (data[start:start + size], len(data))):
        result = inspect_tar("https://example.org/data.tar")
    assert result["files"][1]["name"] == "x" * 120 + ".cool.gz"


def test_tar_inspector_refuses_unbounded_entry_walk():
    data = _archive()
    with patch.dict(inspect_tar.__globals__, read_range=lambda url, start, size: (data[start:start + size], len(data))), pytest.raises(ValueError, match="max_entries"):
        inspect_tar("https://example.org/data.tar", max_entries=1)


def test_range_reader_refuses_full_download_response():
    class Response(io.BytesIO):
        status = 200
        headers: ClassVar[dict] = {}

        def read(self, size=-1):
            raise AssertionError("must reject response before reading full archive")

    with patch.dict(read_range.__globals__, urlopen=lambda *args, **kwargs: Response()), pytest.raises(ValueError, match="refusing full archive"):
        read_range("https://example.org/data.tar", 0, 512)


def test_range_reader_validates_content_range_and_length():
    class Response(io.BytesIO):
        status = 206
        headers: ClassVar[dict] = {"Content-Range": "bytes 0-511/4096"}

    with patch.dict(read_range.__globals__, urlopen=lambda *args, **kwargs: Response(bytes(512))):
        assert read_range("https://example.org/data.tar", 0, 512) == (bytes(512), 4096)
    with patch.dict(read_range.__globals__, urlopen=lambda *args, **kwargs: Response(bytes(511))), pytest.raises(ValueError, match="truncated"):
        read_range("https://example.org/data.tar", 0, 512)
