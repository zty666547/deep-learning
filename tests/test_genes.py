import gzip

import pytest

from microc_foundation.genes import read_genes_gff


def test_gff_coordinates_and_circular_origin_split(tmp_path):
    path = tmp_path / "genes.gff.gz"
    content = (
        "##gff-version 3\n"
        "##sequence-region chrM 1 50\n"
        "chrM\tRefSeq\tregion\t1\t50\t.\t+\t.\tID=chrM;Is_circular=true\n"
        "chrM\tRefSeq\tgene\t46\t55\t.\t+\t.\tID=gene-a;Name=origin_gene\n"
    )
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(content)

    genes = read_genes_gff(str(path), {"chrM": 50})
    assert list(zip(genes.start, genes.end, genes.part)) == [(0, 5, 2), (45, 50, 1)]
    assert genes.gene_name.tolist() == ["origin_gene", "origin_gene"]


def test_gff_reference_mismatch_is_rejected(tmp_path):
    path = tmp_path / "genes.gff"
    path.write_text("##sequence-region chr1 1 100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lengths do not match"):
        read_genes_gff(str(path), {"chr1": 101})
