import cooler
import pandas as pd
import pytest

from microc_foundation.condition_tracks import circular_contact_track, genome_tiles
from microc_foundation.io import open_cooler


def test_genome_tiles_cover_remainder_without_gap():
    assert genome_tiles(25, tile_bp=10) == [(0, 10), (10, 20), (20, 25)]


def test_circular_track_counts_both_genome_ends(tmp_path):
    path = tmp_path / "circular.cool"
    bins = pd.DataFrame({"chrom": ["chr1"] * 5,
                         "start": [0, 10, 20, 30, 40],
                         "end": [10, 20, 30, 40, 50]})
    pixels = pd.DataFrame({"bin1_id": [0, 0, 1, 2], "bin2_id": [0, 1, 2, 3],
                           "count": [1, 2, 5, 7]})
    cooler.create_cooler(str(path), bins, pixels)
    with open_cooler(str(path)) as contact_map:
        track = circular_contact_track(contact_map, band_bp=10, chunk_bins=2)
    assert track.band_contacts.tolist() == [2, 7, 12, 7, 0]
    assert track.possible_partner_bins.tolist() == [2, 2, 2, 2, 2]


def test_rejects_fewer_than_three_conditions():
    from microc_foundation.condition_tracks import build_condition_tracks

    with pytest.raises(ValueError, match="three conditions"):
        build_condition_tracks({"WT": ["a.cool", "b.cool"], "mutant": ["c.cool", "d.cool"]})
