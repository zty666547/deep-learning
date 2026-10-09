"""List an uncompressed HTTP TAR using range reads, without fetching payloads."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from urllib.request import Request, urlopen


def read_range(url: str, start: int, size: int) -> tuple[bytes, int]:
    if start < 0 or size <= 0 or size > 1024 * 1024:
        raise ValueError("metadata range must be between 1 byte and 1 MiB")
    request = Request(url, headers={"Range": f"bytes={start}-{start + size - 1}"})
    with urlopen(request, timeout=20) as response:
        if response.status != 206:
            raise ValueError("server did not honor HTTP Range; refusing full archive download")
        content_range = response.headers.get("Content-Range", "")
        prefix = f"bytes {start}-{start + size - 1}/"
        if not content_range.startswith(prefix):
            raise ValueError("unexpected HTTP Content-Range")
        data = response.read(size + 1)
        if len(data) != size:
            raise ValueError("truncated or oversized metadata response")
        return data, int(content_range.split("/")[-1])


def inspect_tar(url: str, *, max_entries: int = 1000) -> dict:
    if max_entries < 1:
        raise ValueError("max_entries must be positive")
    rows = []
    offset, transferred, total = 0, 0, None
    extended_name = None
    for _ in range(max_entries):
        header, size = read_range(url, offset, 512)
        transferred += len(header)
        if total is not None and size != total:
            raise ValueError("remote archive size changed during inspection")
        total = size
        if header == bytes(512):
            return {"url": url, "archive_size_bytes": total, "metadata_bytes_read": transferred,
                    "num_files": len(rows), "files": rows, "complete": True}
        info = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
        next_offset = offset + 512 + ((info.size + 511) // 512) * 512
        if next_offset > total:
            raise ValueError("member extends past archive size")
        if info.type == tarfile.GNUTYPE_LONGNAME:
            payload, payload_total = read_range(url, offset + 512, info.size)
            if payload_total != total:
                raise ValueError("remote archive size changed during inspection")
            transferred += len(payload)
            extended_name = payload.rstrip(b"\0").decode("utf-8")
        elif info.type in (tarfile.XHDTYPE, tarfile.XGLTYPE):
            raise ValueError("PAX metadata not supported by this lightweight inspector")
        elif info.isfile():
            name = extended_name or info.name
            extended_name = None
            rows.append({"name": name, "size_bytes": info.size,
                         "header_offset": offset, "data_offset": offset + 512})
            print(f"{name}\t{info.size}", flush=True)
        elif not info.isdir():
            raise ValueError("unsupported non-regular TAR entry")
        offset = next_offset
    raise ValueError("max_entries reached before archive terminator")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = inspect_tar(args.url)
    # Output is metadata only; never extract or execute archive members.
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Listed {result['num_files']} files using {result['metadata_bytes_read']} metadata bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
