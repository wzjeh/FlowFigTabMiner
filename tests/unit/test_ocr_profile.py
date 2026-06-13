"""
Unit tests for the OCR profiler (issue #17 Phase 1 instrumentation).

Pure logic, no models — verifies the profiler is a true no-op when disabled
(so production behaviour is byte-identical) and that, when enabled, it records
per-loop crop counts/sizes, attributes OCR sub-timings to the active loop, and
emits a report with all required fields.
"""
import os
from unittest.mock import patch

from src.extraction.common.ocr_profile import _OCRProfiler


def _enabled():
    with patch.dict(os.environ, {"OCR_PROFILE": "1"}):
        return _OCRProfiler()


def _disabled():
    with patch.dict(os.environ, {"OCR_PROFILE": "0"}):
        return _OCRProfiler()


class TestDisabledIsNoOp:
    def test_disabled_records_nothing_and_prints_nothing(self, capsys):
        p = _disabled()
        assert p.enabled is False
        p.start_loop("caption")
        p.record_crop("caption", (10, 20), (30, 60))
        with p.ocr_phase("ocr_inference"):
            pass
        with p.bucket("vlm"):
            pass
        p.end_loop()
        p.report(table="t")
        # No output, no accumulation.
        assert capsys.readouterr().out == ""
        assert p._loops == {}
        assert p._buckets == {}


class TestEnabledRecording:
    def test_record_crop_counts_and_sizes(self):
        p = _enabled()
        p.start_loop("caption")
        p.record_crop("caption", (100, 20), (300, 60))
        p.record_crop("caption", (200, 40), (600, 120))
        p.end_loop()
        d = p._loops["caption"]
        assert d["crops"] == 2
        assert d["input"] == [(100, 20), (200, 40)]
        assert d["resized"] == [(300, 60), (600, 120)]

    def test_ocr_phase_attributes_to_active_loop(self):
        p = _enabled()
        p.start_loop("note")
        with p.ocr_phase("ocr_inference"):
            pass
        with p.ocr_phase("ocr_postprocess"):
            pass
        p.end_loop()
        assert p._loops["note"]["ocr_inference"] >= 0.0
        assert p._loops["note"]["ocr_postprocess"] >= 0.0
        assert "ocr_inference" in p._loops["note"]

    def test_ocr_phase_outside_loop_lands_in_other(self):
        p = _enabled()
        # No start_loop -> current is None -> "_other" bucket.
        with p.ocr_phase("ocr_inference"):
            pass
        assert "_other" in p._loops

    def test_bucket_accumulates(self):
        p = _enabled()
        with p.bucket("serialize"):
            pass
        with p.bucket("serialize"):
            pass
        assert "serialize" in p._buckets
        assert p._buckets["serialize"] >= 0.0

    def test_report_has_all_fields_and_resets(self, capsys):
        p = _enabled()
        p.start_loop("caption")
        p.record_crop("caption", (949, 116), (2947, 448))
        with p.ocr_phase("ocr_inference"):
            pass
        p.end_loop()
        with p.bucket("vlm"):
            pass
        with p.bucket("serialize"):
            pass
        p.report(table="smoke")
        out = capsys.readouterr().out
        # Required fields present in the emitted report.
        for token in ["[OCR_PROFILE]", "table=smoke", "caption:", "crops=1",
                      "ocr_inf=", "ocr_post=", "input p50=", "resized p50=",
                      "vlm:", "serialize:"]:
            assert token in out, f"missing {token!r} in report:\n{out}"
        # report() resets state for the next table.
        assert p._loops == {}
        assert p._buckets == {}

    def test_rep_size_picks_area_percentile(self):
        sizes = [(10, 10), (100, 100), (50, 50)]  # areas 100, 10000, 2500
        # max by area -> 100x100; p50 -> middle area 2500 -> 50x50
        assert _OCRProfiler._rep_size(sizes, 100) == "100x100"
        assert _OCRProfiler._rep_size(sizes, 50) == "50x50"
        assert _OCRProfiler._rep_size([], 50) == "-"
