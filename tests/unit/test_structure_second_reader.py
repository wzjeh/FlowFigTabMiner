"""Suspicious MolNexTR readings, and the VLM reading that may replace them."""
import numpy as np

from src.extraction.common.structure_plausibility import suspicious_smiles
from src.extraction.common.structure_second_reader import VLMStructureReader


def test_suspicious_readings_are_named_by_their_symptom():
    assert suspicious_smiles("Brc1ccc([Na])cc1") == "odd:Na"
    assert suspicious_smiles("C[SH]([Mo])CCl") == "odd:Mo"
    assert suspicious_smiles("CBr.[I-].c1ccc(-c2ccccc2)cc1") == "shards"
    assert suspicious_smiles("C[CH3]C1(c2ccccc2)CO1") == "invalid"
    assert suspicious_smiles("") == "invalid" and suspicious_smiles(None) == "invalid"
    for fine in ("CC1(c2ccccc2)CO1", "*OC(=O)c1ccccc1", "[Li]C(F)I", "CCCC[Sn](CCCC)(CCCC)C(F)I", "CCO.CC(=O)O"):
        assert suspicious_smiles(fine) is None, fine


class _VLM:
    def __init__(self, smiles):
        self.smiles, self.calls = smiles, 0

    def inspect(self, image, system_prompt, user_prompt, cfg, response_schema):
        self.calls += 1
        return object(), {"smiles": self.smiles}


def _crop():
    return np.full((120, 160, 3), 255, dtype=np.uint8)


def test_reader_accepts_only_a_plausible_molecule(tmp_path):
    r = VLMStructureReader(_VLM("N#Cc1ccc(Br)cc1"), cfg=None, scratch_dir=str(tmp_path))
    assert r.read(_crop()) == "N#Cc1ccc(Br)cc1" and r.n_calls == 1 and r.n_accepted == 1
    for bad in ("", "Brc1ccc([Na])cc1", "not smiles", "CBr.[I-].c1ccc(-c2ccccc2)cc1"):
        r = VLMStructureReader(_VLM(bad), cfg=None, scratch_dir=str(tmp_path))
        assert r.read(_crop()) is None and r.n_accepted == 0


def test_reader_survives_a_failing_provider(tmp_path):
    class _Boom:
        def inspect(self, **kw):
            raise RuntimeError("503")
    assert VLMStructureReader(_Boom(), cfg=None, scratch_dir=str(tmp_path)).read(_crop()) is None
