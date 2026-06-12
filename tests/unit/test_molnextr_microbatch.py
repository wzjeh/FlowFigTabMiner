"""
Unit tests for the MolNexTR micro-batch path (issue #14).

These tests NEVER load a real model — MolNexTR is always a Mock — so they are
safe to run on any machine (the real model load is what kernel-panicked the
16GB dev Mac).  They cover ContentRecognizer.recognize_structures_batch:
alignment, chunking, MPS-cache release, invalid-crop handling, and the
per-box retry that protects a table from one malformed crop.
"""
import numpy as np
from unittest.mock import MagicMock, patch

from src.extraction.common.content_recognizer import ContentRecognizer


def _bare_recognizer(molnextr=None):
    """ContentRecognizer without running __init__ (no OCR / MolNexTR load)."""
    rec = ContentRecognizer.__new__(ContentRecognizer)
    rec.molnextr = molnextr
    return rec


def _rgb_crop():
    """A valid 3-channel image array (so _to_rgb passes through cvtColor)."""
    return np.zeros((20, 20, 3), dtype=np.uint8)


def _mol_returning(smiles_by_call):
    """Mock MolNexTR whose predict_images returns dicts with given SMILES.

    smiles_by_call: callable(list_of_images) -> list[str].
    """
    mol = MagicMock()

    def _predict(images, **kwargs):
        return [{"predicted_smiles": s} for s in smiles_by_call(images)]

    mol.predict_images.side_effect = _predict
    return mol


class TestRecognizeStructuresBatch:
    def test_missing_model_returns_sentinel(self):
        rec = _bare_recognizer(molnextr=None)
        out = rec.recognize_structures_batch([_rgb_crop(), _rgb_crop()])
        assert out == ["[MolNexTR Missing]", "[MolNexTR Missing]"]

    def test_alignment_one_to_one(self):
        crops = [_rgb_crop(), _rgb_crop(), _rgb_crop()]
        # Return a deterministic SMILES per crop, in order.
        seq = iter(["CCO", "CCC", "c1ccccc1"])
        mol = _mol_returning(lambda imgs: [next(seq) for _ in imgs])
        rec = _bare_recognizer(molnextr=mol)

        out = rec.recognize_structures_batch(crops, batch_size=8)
        assert out == ["CCO", "CCC", "c1ccccc1"]

    def test_chunking_calls_predict_per_chunk(self):
        crops = [_rgb_crop() for _ in range(5)]
        mol = _mol_returning(lambda imgs: ["X"] * len(imgs))
        rec = _bare_recognizer(molnextr=mol)

        with patch.object(rec, "_mps_empty_cache") as mps:
            out = rec.recognize_structures_batch(crops, batch_size=2)

        # 5 crops / bs=2 -> chunks of [2, 2, 1] = 3 predict_images calls,
        # and the MPS cache is freed once per chunk.
        assert len(out) == 5
        assert mol.predict_images.call_count == 3
        assert mps.call_count == 3

    def test_invalid_crop_stays_empty_and_keeps_alignment(self):
        # Middle input is an unreadable path -> _to_rgb returns None.
        crops = [_rgb_crop(), "/no/such/file.png", _rgb_crop()]
        # predict_images only ever sees the 2 valid crops.
        mol = _mol_returning(lambda imgs: ["A", "B"][: len(imgs)])
        rec = _bare_recognizer(molnextr=mol)

        out = rec.recognize_structures_batch(crops, batch_size=8)
        assert out == ["A", "", "B"]
        # The model was handed only the 2 valid crops.
        (called_imgs,), _ = mol.predict_images.call_args
        assert len(called_imgs) == 2

    def test_chunk_failure_falls_back_to_per_box(self):
        crops = [_rgb_crop(), _rgb_crop(), _rgb_crop()]
        mol = MagicMock()

        def _predict(images, **kwargs):
            if len(images) > 1:
                raise RuntimeError("simulated batch decode failure")
            # per-box retry path
            return [{"predicted_smiles": "RETRY"}]

        mol.predict_images.side_effect = _predict
        rec = _bare_recognizer(molnextr=mol)

        out = rec.recognize_structures_batch(crops, batch_size=8)
        # One bad batch must not blank the table — every box recovered per-box.
        assert out == ["RETRY", "RETRY", "RETRY"]

    def test_empty_input(self):
        mol = _mol_returning(lambda imgs: [])
        rec = _bare_recognizer(molnextr=mol)
        assert rec.recognize_structures_batch([]) == []
        mol.predict_images.assert_not_called()
