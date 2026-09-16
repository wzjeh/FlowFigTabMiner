"""PerSourceAssembler — fan-out LLM calls, one per ``SourcePacket``.

This is the architectural fix for the 32k output-token cap that was
truncating GlobalAssembly mid-record.  Instead of one giant call asking
Gemini to emit every reaction record across the whole paper in a single
JSON array, we ask one call per figure / table.  Each call emits 5–20
records, well within the budget.

Concurrency: ``ThreadPoolExecutor`` fan-out, but the real rate gate is
the module-level ``acquire_llm_slot()`` semaphore inside
``GeminiProvider``.  Oversubscribing the pool (10 workers vs 5 slots)
just keeps the slots saturated; it does not actually exceed
``llm.max_concurrent``.

Failure isolation: a single source's parse / call failure must NOT kill
the whole PDF — log it, save the raw response for forensics, return an
empty list for that source, keep all others.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Mapping

from src.adjudication.figure_synthesis import synthesize_records, validate_template
from src.adjudication.per_source_prompts import (
    CommonPreamble,
    FigureTemplateBuilder,
    PerSourcePromptBuilder,
)
from src.adjudication.source_discovery import SourcePacket
from src.llm.config import LLMConfig
from src.llm.json_utils import sanitize_json_text
from src.llm.providers.base import LLMProvider
from src.llm.types import ChatMessage, Role
from src.pipeline.status import write_status

logger = logging.getLogger(__name__)


class PerSourceAssembler:
    """Fan-out LLM calls across all sources in a PDF; aggregate records."""

    def __init__(
        self,
        llm: LLMProvider,
        llm_cfg: LLMConfig,
        prompt_builders: Mapping[str, PerSourcePromptBuilder],
        max_workers: int = 10,
        raw_dir: str = "data/intermediate",
        figure_synthesis: bool = True,
    ):
        self.llm = llm
        self.llm_cfg = llm_cfg
        self.prompt_builders = dict(prompt_builders)
        self.max_workers = max_workers
        self.raw_dir = raw_dir
        # Design step D: figures go through ONE template call + code synthesis;
        # the legacy N-record transcription remains as the fallback.
        self.figure_synthesis = figure_synthesis
        self.template_builder = FigureTemplateBuilder()

    def assemble(
        self,
        packets: List[SourcePacket],
        preamble: CommonPreamble,
        basename: str,
    ) -> List[Dict[str, Any]]:
        """Run one LLM call per packet in parallel; merge into one record list.

        Records get tagged with ``source_table_or_figure = packet.human_label``
        and ``__assembly_order`` (stripped after sort) so the final list is
        deterministic regardless of completion order.
        """
        if not packets:
            logger.info("per_source.assemble no packets — empty result")
            return []

        raw_dir = os.path.join(self.raw_dir, basename, "per_source_raw")
        os.makedirs(raw_dir, exist_ok=True)

        wall_t0 = time.perf_counter()
        successful = 0
        failed = 0
        total_retries = 0
        all_records: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._run_one, pkt, preamble, raw_dir, idx): pkt
                for idx, pkt in enumerate(packets)
            }
            for fut in as_completed(futures):
                pkt = futures[fut]
                try:
                    recs, retries = fut.result()
                    all_records.extend(recs)
                    total_retries += retries
                    if recs:
                        successful += 1
                    else:
                        # Parse failure or empty array — counted as failed
                        # only if we actually called and got no usable JSON.
                        failed += 1
                except Exception as exc:
                    logger.exception("per_source.fail source=%s exc=%s", pkt.source_id, exc)
                    failed += 1

        # Deterministic order: by human label, then assembly order.
        all_records.sort(key=lambda r: (
            str(r.get("source_table_or_figure", "")),
            int(r.pop("__assembly_order", 0)),
        ))

        wall_ms = (time.perf_counter() - wall_t0) * 1000.0
        logger.info(
            "per_source.summary basename=%s sources=%d successful=%d failed=%d "
            "records=%d total_retries=%d wall_ms=%.0f",
            basename, len(packets), successful, failed, len(all_records),
            total_retries, wall_ms,
        )
        return all_records

    # ── internals ──────────────────────────────────────────────────────

    def _run_figure_template(self, packet, preamble, raw_dir, order_idx):
        """One template call + deterministic synthesis.  Returns
        ``(records, retries)`` or ``(None, 0)`` when the template is unusable
        (caller falls back to the legacy path)."""
        raw_data = packet.evidence.get("raw_data", []) or []
        if not raw_data:
            return [], 0
        try:
            system_prompt, user_prompt = self.template_builder.build(packet, preamble)
            response = self.llm.chat(
                [ChatMessage(role=Role.SYSTEM, content=system_prompt),
                 ChatMessage(role=Role.USER, content=user_prompt)],
                self.llm_cfg,
            )
        except Exception as exc:
            logger.error("per_source.template_llm_fail source=%s exc=%s", packet.source_id, exc)
            return None, 0
        raw_text = response.text or ""
        try:
            with open(os.path.join(raw_dir, f"{packet.source_id}_template_raw.txt"), "w") as f:
                f.write(raw_text)
        except Exception:
            pass
        try:
            tpl = json.loads(sanitize_json_text(raw_text))
        except Exception as exc:
            logger.error("per_source.template_parse_fail source=%s exc=%s", packet.source_id, exc)
            return None, 0
        if isinstance(tpl, list) and len(tpl) == 1 and isinstance(tpl[0], dict):
            tpl = tpl[0]
        err = validate_template(tpl)
        if err:
            logger.error("per_source.template_invalid source=%s err=%s", packet.source_id, err)
            return None, 0
        facts = (packet.evidence.get("meta", {}) or {}).get("facts") or {}
        records = synthesize_records(tpl, raw_data, packet.human_label, facts)
        for j, rec in enumerate(records):
            rec["__source_id"] = packet.source_id
            rec["__assembly_order"] = order_idx * 1000 + j
        retry_count = getattr(response, "retry_count", 0) or 0
        write_status(os.path.dirname(raw_dir), packet.source_id, "assembly", "ok", "template+synthesis",
                     records=len(records), synthesized=True)
        logger.info(
            "per_source.template_ok source=%s records=%d latency_ms=%.0f tokens_out=%s axis_map=%s",
            packet.source_id, len(records), response.latency_ms, response.tokens_out,
            json.dumps(tpl.get("axis_map")),
        )
        return records, retry_count

    def _run_one(
        self,
        packet: SourcePacket,
        preamble: CommonPreamble,
        raw_dir: str,
        order_idx: int,
    ) -> tuple[List[Dict[str, Any]], int]:
        """Execute one source's LLM call; return ``(records, retry_count)``.

        ``retry_count`` reflects how many times the provider's exponential
        backoff fired before the call succeeded (0 = first-try OK).
        """
        builder = self.prompt_builders.get(packet.source_type)
        if builder is None:
            logger.warning(
                "per_source.no_builder source=%s type=%s — skipping",
                packet.source_id, packet.source_type,
            )
            return [], 0

        if packet.source_type == "figure" and self.figure_synthesis:
            recs, retries = self._run_figure_template(packet, preamble, raw_dir, order_idx)
            if recs is not None:
                return recs, retries
            logger.warning("per_source.template_fallback source=%s — using legacy transcription", packet.source_id)

        try:
            system_prompt, user_prompt = builder.build(packet, preamble)
        except Exception as exc:
            logger.exception("per_source.prompt_build_fail source=%s exc=%s", packet.source_id, exc)
            return [], 0

        try:
            response = self.llm.chat(
                [
                    ChatMessage(role=Role.SYSTEM, content=system_prompt),
                    ChatMessage(role=Role.USER, content=user_prompt),
                ],
                self.llm_cfg,
            )
        except Exception as exc:
            logger.error("per_source.llm_fail source=%s exc=%s", packet.source_id, exc)
            write_status(os.path.dirname(raw_dir), packet.source_id, "assembly", "failed", f"LLM call failed: {exc}")
            return [], 0

        raw_text = response.text or ""

        # Forensic raw dump — always.
        raw_path = os.path.join(raw_dir, f"{packet.source_id}_raw.txt")
        try:
            with open(raw_path, "w") as f:
                f.write(raw_text)
        except Exception:
            logger.warning("per_source.raw_write_fail path=%s", raw_path)

        cleaned = sanitize_json_text(raw_text)
        try:
            parsed = json.loads(cleaned)
        except Exception as exc:
            logger.error(
                "per_source.parse_fail source=%s exc=%s raw=%s",
                packet.source_id, exc, raw_path,
            )
            write_status(os.path.dirname(raw_dir), packet.source_id, "assembly", "failed", f"JSON parse failed: {exc}")
            return [], 0

        if not isinstance(parsed, list):
            logger.error(
                "per_source.shape_fail source=%s got=%s raw=%s",
                packet.source_id, type(parsed).__name__, raw_path,
            )
            return [], 0

        # Belt-and-braces tagging + ordering aid.
        records: List[Dict[str, Any]] = []
        for j, rec in enumerate(parsed):
            if not isinstance(rec, dict):
                continue  # skip non-object entries
            rec.setdefault("source_table_or_figure", packet.human_label)
            rec["__source_id"] = packet.source_id
            rec["__assembly_order"] = order_idx * 1000 + j
            records.append(rec)

        retry_count = getattr(response, "retry_count", 0) or 0
        write_status(os.path.dirname(raw_dir), packet.source_id, "assembly", "ok", "", records=len(records))
        logger.info(
            "per_source.ok source=%s type=%s records=%d latency_ms=%.0f tokens_out=%s retry=%d",
            packet.source_id, packet.source_type, len(records),
            response.latency_ms, response.tokens_out, retry_count,
        )
        return records, retry_count
