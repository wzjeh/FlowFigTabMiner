"""Step 3.5b: labels printed next to structures in figure schemes -> compound pool."""
import json

from src.extraction.scheme.figure_scheme_harvest import (CompoundReading, RVariant, filtered_scheme_crops, label_cells,
                                                         readings_to_pool, substitute_r)


def test_r_group_substitution_expands_a_wildcard_core():
    assert substitute_r("*C1OC1(C)c1ccccc1", "Me") == "CC1OC1(C)c1ccccc1"
    assert substitute_r("*C1OC1(C)c1ccccc1", "Ph") == "CC1(c2ccccc2)OC1c1ccccc1"
    assert substitute_r("*OC(=O)c1ccccc1", "tert-butyl") == "CC(C)(C)OC(=O)c1ccccc1"
    assert substitute_r("*OC(=O)c1ccccc1", "H") == "O=C(O)c1ccccc1"
    assert substitute_r("*C(*)C", "Me") is None            # two attachment points: ambiguous
    assert substitute_r("CCO", "Me") is None               # nothing to substitute
    assert substitute_r("*C1OC1(C)c1ccccc1", "Xyl") is None  # unknown group


def test_readings_pair_with_structures_and_expand_families():
    mols = [{"smiles": "*C1OC1(C)c1ccccc1"}, {"smiles": "CC1(c2ccccc2)CO1"}, {"smiles": "*c1ccccc1"}, {"smiles": "C[CH3]C1(c2ccccc2)CO1"}]
    readings = [CompoundReading(index=0, variants=[RVariant(label="c-9", r_group="Me"), RVariant(label="c-11", r_group="Ph")]),
                CompoundReading(index=1, label="3"), CompoundReading(index=2, label="4"),      # wildcard without R table: skipped
                CompoundReading(index=3, label="5"),                                           # MolNexTR misread RDKit rejects: skipped
                CompoundReading(index=9, label="x")]
    assert readings_to_pool(mols, readings) == {"c-9": "CC1OC1(C)c1ccccc1", "c-11": "CC1(c2ccccc2)OC1c1ccccc1", "3": "CC1(c2ccccc2)CO1"}


def test_label_cells_extend_below_the_structure_and_stay_inside():
    cells = label_cells([{"box": [100, 100, 200, 150]}], (160, 300, 3))
    assert cells[0]["box"] == [75, 92.5, 225, 160]


def test_filtered_scheme_crops_follow_the_status_files(tmp_path):
    (tmp_path / "status").mkdir(); (tmp_path / "figures").mkdir()
    json.dump({"source_id": "page_2_figure_0", "stage": "macro_clean", "outcome": "filtered"}, open(tmp_path / "status" / "page_2_figure_0.json", "w"))
    json.dump({"source_id": "page_3_figure_0", "stage": "assembly", "outcome": "ok"}, open(tmp_path / "status" / "page_3_figure_0.json", "w"))
    json.dump({"source_id": "page_9_figure_0", "stage": "macro_clean", "outcome": "filtered"}, open(tmp_path / "status" / "page_9_figure_0.json", "w"))
    (tmp_path / "figures" / "page_2_figure_0.png").write_bytes(b"x")
    assert [p.split("/")[-1] for p in filtered_scheme_crops(str(tmp_path))] == ["page_2_figure_0.png"]   # a chart and a missing crop are left out
