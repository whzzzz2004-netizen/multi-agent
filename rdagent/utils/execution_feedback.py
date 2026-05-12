"""Heuristics for classifying implementation feedback (environment vs code)."""

from __future__ import annotations

import re
from typing import Any

_ENV_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"pip install.*non-zero", re.I),
    re.compile(r"returned a non-zero code", re.I),
    re.compile(r"docker build.*fail", re.I),
    re.compile(r"error building docker", re.I),
    re.compile(r"No space left on device", re.I),
    re.compile(r"Connection timed out", re.I),
    re.compile(r"Could not find a version that satisfies", re.I),
    re.compile(r"Temporary failure in name resolution", re.I),
    re.compile(r"Read timed out", re.I),
)


def execution_text_suggests_environment_error(*texts: str | None) -> bool:
    blob = " ".join(t for t in texts if t)
    return any(p.search(blob) for p in _ENV_PATTERNS)


def coosteer_multifeedback_all_environment_errors(feedback: Any) -> bool:
    """
    True when every non-None sub-feedback execution text matches infra/pip/docker heuristics.
    Duck-typed for CoSTEERMultiFeedback without importing CoSTEER from core.
    """
    fl = getattr(feedback, "feedback_list", None)
    if not isinstance(fl, list) or not fl:
        return False
    for sub in fl:
        if sub is None:
            continue
        ex = getattr(sub, "execution", None) or ""
        rc = getattr(sub, "return_checking", None) or ""
        if not execution_text_suggests_environment_error(ex, rc):
            return False
    return True
