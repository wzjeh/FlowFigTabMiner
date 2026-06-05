"""Core ABCs and typed results for the pipeline-internal fusion layer."""

from __future__ import annotations

import abc
import enum
import logging
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Severity(str, enum.Enum):
    """Severity of an issue raised by a ``ConsistencyCheck``."""

    INFO = "info"        # advisory only, no confidence penalty
    WARNING = "warning"  # small penalty (e.g. legend color too close)
    ERROR = "error"      # large penalty (e.g. RANSAC inlier ratio < 0.7)


class Issue(BaseModel):
    """One disagreement spotted by a check."""

    check: str            # short id, e.g. "point_count_consistency"
    severity: Severity
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


class ConsistencyResult(BaseModel):
    """Aggregated output of one or more ``ConsistencyCheck.run`` calls.

    ``confidence_delta`` is the suggested adjustment to apply to a stage's
    overall confidence: each check contributes a number in [-1, 0]; the
    aggregator sums or composes them as the consuming pipeline sees fit.
    """

    issues: list[Issue] = Field(default_factory=list)
    confidence_delta: float = 0.0   # in [-1, 0], where 0 = no change

    def merge(self, other: "ConsistencyResult") -> "ConsistencyResult":
        return ConsistencyResult(
            issues=[*self.issues, *other.issues],
            confidence_delta=self.confidence_delta + other.confidence_delta,
        )

    @property
    def has_errors(self) -> bool:
        return any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == Severity.WARNING for i in self.issues)


class ConsistencyCheck(abc.ABC):
    """Abstract base for one cross-source consistency check.

    Concrete checks live in ``figure_checks.py`` / ``table_checks.py`` and
    take whatever subset of ``stage_outputs`` they need (each check
    declares the keys it expects via ``required_keys``).
    """

    #: Short stable identifier used in ``Issue.check`` and logging.
    id: str = "consistency_check"

    #: Keys this check expects to find in the ``stage_outputs`` dict.
    required_keys: tuple[str, ...] = ()

    @abc.abstractmethod
    def run(self, stage_outputs: dict[str, Any]) -> ConsistencyResult:
        """Run the check; return the issues + confidence adjustment."""

    def _missing_keys(self, stage_outputs: dict[str, Any]) -> Optional[Issue]:
        missing = [k for k in self.required_keys if k not in stage_outputs]
        if not missing:
            return None
        return Issue(
            check=self.id,
            severity=Severity.INFO,
            message=f"check skipped, missing stage outputs: {missing}",
            detail={"missing": missing},
        )


def run_checks(
    checks: Iterable[ConsistencyCheck],
    stage_outputs: dict[str, Any],
    *,
    stage_tag: str = "?",
    source_id: str = "?",
) -> ConsistencyResult:
    """Run a batch of checks against a stage's outputs and aggregate.

    A single canonical entry point so pipeline callers don't write their
    own loops.  Every check execution is logged with its inputs (a
    digest of the relevant keys), result severity, and confidence delta.

    The ``stage_tag`` and ``source_id`` tags make it trivial to grep
    pipeline logs after a run::

        grep "fusion.run_checks" pipeline.log | grep "fig3a"
    """
    aggregate = ConsistencyResult()
    for check in checks:
        # Project ``stage_outputs`` to the subset this check looks at so
        # the log line is concise even when callers pass big dicts.
        sample = {k: stage_outputs.get(k) for k in check.required_keys}
        result = check.run(stage_outputs)
        aggregate = aggregate.merge(result)
        severities = [i.severity.value for i in result.issues] or ["clean"]
        logger.info(
            "fusion.run_checks stage=%s source=%s check=%s inputs=%s severities=%s conf_delta=%+.3f",
            stage_tag,
            source_id,
            check.id,
            sample,
            severities,
            result.confidence_delta,
        )
        for issue in result.issues:
            # One log line per non-clean issue at matching severity,
            # carrying the message so audit trail tells "why" not just "what".
            level = {
                Severity.INFO: logging.INFO,
                Severity.WARNING: logging.WARNING,
                Severity.ERROR: logging.ERROR,
            }[issue.severity]
            logger.log(
                level,
                "fusion.issue stage=%s source=%s check=%s severity=%s msg=%s detail=%s",
                stage_tag,
                source_id,
                issue.check,
                issue.severity.value,
                issue.message,
                issue.detail,
            )

    logger.info(
        "fusion.run_checks.summary stage=%s source=%s checks=%d issues=%d total_conf_delta=%+.3f errors=%s warnings=%s",
        stage_tag,
        source_id,
        sum(1 for _ in checks) if not isinstance(checks, list) else len(checks),
        len(aggregate.issues),
        aggregate.confidence_delta,
        aggregate.has_errors,
        aggregate.has_warnings,
    )
    return aggregate
