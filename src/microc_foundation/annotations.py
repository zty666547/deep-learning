"""Validated readers for known chromatin-structure annotations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_ALIASES: dict[str, str] = {
    "chrom": "chrom",
    "chr": "chrom",
    "chromosome": "chrom",
    "start": "start",
    "end": "end",
    "structure_type": "structure_type",
    "type": "structure_type",
    "label": "structure_type",
    "class": "structure_type",
}

_STRUCTURE_SHEETS: dict[str, str] = {
    "Supplementary Table 4": "OPCID",
    "Supplementary Table 5": "CHIN",
    "Supplementary Table 6": "CHID",
}


def _validate_structures_frame(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Validate a normalized structure table without dropping extra columns."""

    required = {"chrom", "start", "end"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError(f"{source_name} contains no rows")

    frame = frame.copy()
    frame["chrom"] = frame["chrom"].astype(str).str.strip()
    for column in ("start", "end"):
        numeric = pd.to_numeric(frame[column], errors="raise")
        if ((numeric % 1) != 0).any():
            raise ValueError(f"{column} coordinates must be integers")
        frame[column] = numeric.astype("int64")

    if (frame["chrom"] == "").any():
        raise ValueError("chrom values must not be empty")
    if (frame["start"] < 0).any():
        raise ValueError("start coordinates must be non-negative")
    if (frame["end"] <= frame["start"]).any():
        raise ValueError("every end coordinate must be greater than start")
    if "structure_type" in frame.columns:
        frame["structure_type"] = frame["structure_type"].astype(str).str.strip()
    return frame


def read_structures_csv(path: str) -> pd.DataFrame:
    """Read and validate ``structures.csv``.

    Required logical columns are ``chrom``, ``start`` and ``end``. Common
    aliases are normalized, while all additional columns are retained. Genomic
    coordinates use the zero-based, half-open convention.
    """

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Structure annotation file not found: {source}")

    frame = pd.read_csv(source)
    frame.columns = [str(column).strip() for column in frame.columns]
    rename = {
        column: _ALIASES[column.lower()]
        for column in frame.columns
        if column.lower() in _ALIASES
        and column != _ALIASES[column.lower()]
        and _ALIASES[column.lower()] not in frame.columns
    }
    frame = frame.rename(columns=rename)

    return _validate_structures_frame(frame, "structures.csv")


def read_structures_excel(
    path: str,
    chrom_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Read OPCID, CHIN and CHID annotations from the supplied workbook.

    The function reads Supplementary Tables 4, 5 and 6, preserves source
    provenance, and returns the same zero-based, half-open coordinate schema
    accepted by :func:`read_structures_csv`. CHIN's supplied center is kept;
    other structure centers are calculated as the integer interval midpoint.
    """

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Structure annotation workbook not found: {source}")

    try:
        sheets = pd.read_excel(source, sheet_name=list(_STRUCTURE_SHEETS))
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Reading .xlsx annotations requires openpyxl; install the project dependencies."
        ) from exc

    normalized_frames: list[pd.DataFrame] = []
    for sheet_name, structure_type in _STRUCTURE_SHEETS.items():
        sheet = sheets[sheet_name].dropna(how="all").copy()
        id_column = f"{structure_type}_ID"
        required = {id_column, "Chr", "Start", "End"}
        missing = sorted(required.difference(sheet.columns))
        if missing:
            raise ValueError(f"{sheet_name} is missing columns: {', '.join(missing)}")

        sheet = sheet.loc[sheet[id_column].notna()].copy()
        output = pd.DataFrame(
            {
                "structure_id": sheet[id_column].astype(str).str.strip(),
                "source_chrom": sheet["Chr"].astype(str).str.strip(),
                "start": sheet["Start"],
                "end": sheet["End"],
                "structure_type": structure_type,
                "source_sheet": sheet_name,
                "redc_signal_wt": sheet.get("RedC signal WT"),
                "redc_signal_wt_hs": sheet.get("RedC signal WT HS"),
            }
        )
        if structure_type == "CHIN" and "Center" in sheet.columns:
            output["center"] = sheet["Center"]
            if "Resonse to H-NS knockout" in sheet.columns:
                output["response_to_hns_knockout"] = sheet[
                    "Resonse to H-NS knockout"
                ]
        else:
            output["center"] = (
                pd.to_numeric(output["start"], errors="raise")
                + pd.to_numeric(output["end"], errors="raise")
            ) // 2
        normalized_frames.append(output)

    frame = pd.concat(normalized_frames, ignore_index=True)
    frame["chrom"] = frame["source_chrom"]
    if chrom_map:
        frame["chrom"] = frame["chrom"].replace(chrom_map)
    frame["center"] = pd.to_numeric(frame["center"], errors="raise").astype("int64")
    columns = [
        "structure_id",
        "chrom",
        "start",
        "center",
        "end",
        "structure_type",
        "source_chrom",
        "source_sheet",
        "redc_signal_wt",
        "redc_signal_wt_hs",
        "response_to_hns_knockout",
    ]
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return _validate_structures_frame(frame[columns], source.name)


def write_structures_csv(frame: pd.DataFrame, output_path: str) -> Path:
    """Validate and write a normalized annotation table."""

    validated = _validate_structures_frame(frame, "structure annotations")
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(output, index=False)
    return output
