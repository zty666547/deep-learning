"""Download selected course samples from GEO, not the entire 4.98 GB TAR."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

from microc_foundation.io import open_cooler

SAMPLES = ("GSM8950761", "GSM8950762", "GSM8950763", "GSM8950764")
EXPECTED_GENOTYPES = {
    "GSM8950761": "genotype: delta stpA",
    "GSM8950762": "genotype: delta stpA",
    "GSM8950763": "genotype: delta HNS delta StpA",
    "GSM8950764": "genotype: delta HNS delta StpA",
}


def sample_url(name: str) -> str:
    sample = name.split("_")[0]
    if sample not in SAMPLES or Path(name).name != name or not name.endswith(".cool.gz"):
        raise ValueError("unexpected sample or unsafe filename")
    return f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{sample[:-3]}nnn/{sample}/suppl/{name}"


def verify_file(path: Path, expected_size: int) -> dict:
    if path.stat().st_size != expected_size:
        raise ValueError(f"size mismatch: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    # Decompression checks gzip CRC/truncation; COOL validates HDF5 metadata.
    with open_cooler(str(path)) as contact_map:
        chroms = {str(k): int(v) for k, v in contact_map.chromsizes.items()}
        if chroms != {"NC_000913.3": 4_641_652} or contact_map.binsize != 10:
            raise ValueError(f"unexpected reference or resolution: {path.name}")
        result = {"sha256": digest.hexdigest(), "chromsizes": chroms,
                  "binsize": int(contact_map.binsize), "nbins": int(contact_map.info["nbins"]),
                  "nnz": int(contact_map.info["nnz"]), "sum": int(contact_map.info["sum"])}
    return result


def download_one(row: dict, directory: Path) -> dict:
    name = row["name"]
    url = sample_url(name)
    target = directory / name
    expected = int(row["size_bytes"])
    if not target.exists():
        partial = directory / f"{name}.part"
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected:
            raise ValueError(f"partial download exceeds expected size: {partial.name}")
        request = Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        with urlopen(request, timeout=60) as response:
            if offset:
                content_range = response.headers.get("Content-Range", "")
                expected_range = f"bytes {offset}-{expected - 1}/{expected}"
                if response.status != 206 or content_range != expected_range:
                    raise ValueError("server did not honor resume range; partial file retained")
                expected_response = expected - offset
            else:
                if response.status != 200:
                    raise ValueError("unexpected download status")
                expected_response = expected
            if int(response.headers.get("Content-Length", -1)) != expected_response:
                raise ValueError("unexpected download status/length")
            received = offset
            with partial.open("ab" if offset else "xb") as output:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    received += len(block)
                    if received > expected:
                        raise ValueError("download exceeds expected size")
                    output.write(block)
                if received != expected:
                    raise ValueError("truncated download; partial file retained for resume")
        # A .part.gz suffix is needed by the common compressed COOL reader.
        # Validate before promoting; no invalid final filename is exposed.
        compressed_check = directory / f"{name}.checking.cool.gz"
        if compressed_check.exists():
            raise FileExistsError(compressed_check)
        partial.rename(compressed_check)
        try:
            verification = verify_file(compressed_check, expected)
        except Exception:
            compressed_check.rename(partial)
            raise
        compressed_check.rename(target)
    else:
        verification = verify_file(target, expected)
    sample = name.split("_")[0]
    metadata_url = (f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={sample}"
                    "&targ=self&form=text&view=full")
    with urlopen(metadata_url, timeout=30) as response:
        metadata = response.read().decode("utf-8")
    metadata = metadata.replace("\r\n", "\n").replace("\r", "\n")
    if f"^SAMPLE = {sample}" not in metadata or "!Sample_characteristics_ch1" not in metadata:
        raise ValueError("GEO sample metadata missing; refusing an unverified condition label")
    if EXPECTED_GENOTYPES[sample] not in metadata:
        raise ValueError(f"GEO genotype does not match selected condition: {sample}")
    metadata_path = directory / f"{sample}.txt"
    if (metadata_path.exists() and
            metadata_path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
            != metadata):
        raise ValueError("existing sample metadata differs; inspect before replacing")
    metadata_path.write_text(metadata, encoding="utf-8")
    characteristics = [line for line in metadata.splitlines()
                       if line.startswith(("!Sample_title", "!Sample_characteristics_ch1"))]
    print(f"Validated {name}: {verification['sum']} contacts", flush=True)
    return {"name": name, "size_bytes": expected, "source_url": url,
            "metadata_url": metadata_url, "characteristics": characteristics,
            "validation": verification,
            "hash_note": "Local download SHA-256; no publisher SHA-256 available for comparison."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.archive_manifest).read_text())
    rows = [row for row in manifest["files"] if row["name"].split("_")[0] in SAMPLES]
    if len(rows) != len(SAMPLES) or len({r["name"].split("_")[0] for r in rows}) != len(SAMPLES):
        raise ValueError("manifest must contain each selected sample exactly once")
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda row: download_one(row, directory), rows))
    receipt = Path(args.receipt)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({"samples": results}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
