"""
Unit tests for MoleculeProcessor.process_image three-phase refactor (issue #14).

No real YOLO or MolNexTR is loaded — the detector and the ContentRecognizer are
both Mocks — so these are safe to run anywhere.  They verify that the
crop / batch-recognise / per-box-fallback split preserves behaviour:
SMILES stay aligned to their boxes, the micro-batch path is used by default,
MOLNEXTR_BATCH=0 rolls back to per-box recognize_content, and an empty SMILES
still triggers the per-box OCR fallback.
"""
import numpy as np
from unittest.mock import MagicMock

from src.extraction.common.molecule_processor import MoleculeProcessor


class _CPUArr:
    """Mimics a torch tensor row: supports .cpu().numpy()."""

    def __init__(self, arr):
        self.arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class _FakeBox:
    def __init__(self, xyxy, conf, cls=0):
        self.xyxy = [_CPUArr(np.array(xyxy, dtype=float))]
        self.conf = [float(conf)]
        self.cls = [int(cls)]


def _make_processor(boxes):
    """A MoleculeProcessor whose YOLO model is a Mock returning `boxes`."""
    proc = MoleculeProcessor(model_path=None)  # model=None branch, no load
    results = MagicMock()
    results.boxes = boxes
    model = MagicMock()
    model.return_value = [results]      # self.model(img, ...)[0]
    model.names = {0: "molecule"}
    proc.model = model
    return proc


def _img():
    return np.full((300, 300, 3), 255, dtype=np.uint8)


def _boxes_3():
    # Three well-separated, in-bounds boxes.
    return [
        _FakeBox([10, 10, 60, 60], 0.9),
        _FakeBox([100, 100, 160, 160], 0.8),
        _FakeBox([200, 200, 260, 260], 0.7),
    ]


class TestProcessImageMicrobatch:
    def test_microbatch_path_aligns_smiles_to_boxes(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)          # keep mol_debug_crops out of repo
        monkeypatch.delenv("MOLNEXTR_BATCH", raising=False)
        proc = _make_processor(_boxes_3())

        cr = MagicMock()
        cr.recognize_structures_batch.return_value = ["CCO", "CCC", "c1ccccc1"]

        _, metrics = proc.process_image(_img(), cr, mask_only=True)

        # Micro-batch used exactly once; per-box recognize_content NOT used here.
        assert cr.recognize_structures_batch.call_count == 1
        crops_arg = cr.recognize_structures_batch.call_args[0][0]
        assert len(crops_arg) == 3
        assert [m["smiles"] for m in metrics] == ["CCO", "CCC", "c1ccccc1"]
        assert [m["box"] for m in metrics] == [
            [10, 10, 60, 60], [100, 100, 160, 160], [200, 200, 260, 260]
        ]

    def test_empty_smiles_triggers_ocr_fallback(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("MOLNEXTR_BATCH", raising=False)
        proc = _make_processor(_boxes_3())

        cr = MagicMock()
        cr.recognize_structures_batch.return_value = ["CCO", "", "c1ccccc1"]
        cr.recognize_content.return_value = "FALLBACK_TEXT"   # OCR fallback

        _, metrics = proc.process_image(_img(), cr, mask_only=True)

        # The blank box (index 1) recovered via the per-box Text OCR fallback.
        assert cr.recognize_content.call_count == 1
        _, kwargs = cr.recognize_content.call_args
        assert cr.recognize_content.call_args[0][1] == "Text"
        assert [m["smiles"] for m in metrics] == ["CCO", "FALLBACK_TEXT", "c1ccccc1"]

    def test_rollback_env_uses_per_box_recognize_content(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("MOLNEXTR_BATCH", "0")
        proc = _make_processor(_boxes_3())

        cr = MagicMock()
        cr.recognize_content.return_value = "CCO"   # used for both Structure + any fallback

        _, metrics = proc.process_image(_img(), cr, mask_only=True)

        # Rollback path: batch never called; recognize_content drives Structure rec.
        cr.recognize_structures_batch.assert_not_called()
        structure_calls = [
            c for c in cr.recognize_content.call_args_list if c[0][1] == "Structure"
        ]
        assert len(structure_calls) == 3
        assert [m["smiles"] for m in metrics] == ["CCO", "CCO", "CCO"]

    def test_no_boxes_returns_empty_metrics(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        proc = _make_processor([])
        cr = MagicMock()
        _, metrics = proc.process_image(_img(), cr, mask_only=True)
        assert metrics == []
        cr.recognize_structures_batch.assert_not_called()
