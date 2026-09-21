import numpy as np

from microc_foundation.visualization import render_heatmap
from microc_foundation.windows import GenomicWindow


def test_render_heatmap_creates_nonempty_png(tmp_path):
    output = tmp_path / "heatmap.png"
    result = render_heatmap(
        np.array([[4.0, 2.0], [2.0, 3.0]]),
        GenomicWindow("chr1", 0, 2000),
        str(output),
    )
    assert result == output
    assert output.read_bytes().startswith(b"\x89PNG")
    assert output.stat().st_size > 0

