"""Input helpers for single-resolution ``.cool`` and ``.cool.gz`` files."""

from __future__ import annotations

import gzip
import shutil
import tempfile
from pathlib import Path
from types import TracebackType
from typing import Optional, Type


class CoolerResource:
    """Context manager that keeps a temporary decompressed COOL file alive.

    A COOL file is HDF5 and therefore cannot be queried directly through a gzip
    stream. For ``.cool.gz`` inputs this class decompresses to a temporary file,
    opens it with :mod:`cooler`, and removes the temporary file on exit.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._temporary_path: Optional[Path] = None
        self.cooler = None

    def __enter__(self):
        try:
            import cooler
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "Missing dependency 'cooler'. Install the project with `pip install -e .`."
            ) from exc

        source_path = self.path
        if self.path.name.endswith(".cool.gz"):
            handle = tempfile.NamedTemporaryFile(suffix=".cool", delete=False)
            self._temporary_path = Path(handle.name)
            try:
                with gzip.open(self.path, "rb") as compressed, handle:
                    shutil.copyfileobj(compressed, handle)
            except Exception:
                self._cleanup()
                raise
            source_path = self._temporary_path

        self.cooler = cooler.Cooler(str(source_path))
        # Force a lightweight read so malformed files fail inside the context.
        _ = self.cooler.info
        return self.cooler

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.cooler = None
        self._cleanup()

    def _cleanup(self) -> None:
        if self._temporary_path is not None:
            self._temporary_path.unlink(missing_ok=True)
            self._temporary_path = None


def open_cooler(path: str) -> CoolerResource:
    """Return a context manager for a local ``.cool`` or ``.cool.gz`` file.

    Parameters
    ----------
    path:
        Path to a single-resolution COOL file. Multi-resolution ``.mcool`` is
        intentionally outside the first-round scope.
    """

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Micro-C contact map not found: {source}")
    if not (source.name.endswith(".cool") or source.name.endswith(".cool.gz")):
        raise ValueError("Expected a .cool or .cool.gz file")
    return CoolerResource(source)

