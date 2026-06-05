import yaml
import os
from typing import TypedDict, Optional


# ── Schema definitions ────────────────────────────────────────────────────────

class _GlobalConfig(TypedDict):
    output_base_dir: str
    intermediate_dir: str
    device: str

class _Step2MacroConfig(TypedDict):
    model_path: str
    confidence_threshold: float
    device: str

class _Step3MicroConfig(TypedDict):
    model_path: str
    crop_padding: int
    ocr_engine: str
    confidence_threshold: float

class _Step4AssemblyConfig(TypedDict):
    clustering_threshold: float

class _FiguresConfig(TypedDict):
    step2_macro: _Step2MacroConfig
    step3_micro: _Step3MicroConfig
    step4_assembly: _Step4AssemblyConfig

class _TablesSegConfig(TypedDict):
    model_path: str
    threshold: float

class _TablesStructureConfig(TypedDict):
    model_path: str

class _TablesContentConfig(TypedDict):
    ocr_engine: str

class _TablesConfig(TypedDict):
    segmentation: _TablesSegConfig
    structure: _TablesStructureConfig
    content: _TablesContentConfig

class _LLMAdjudicationConfig(TypedDict):
    model_name: str
    temperature: float

class _LLMConfig(TypedDict):
    default_provider: str
    adjudication: _LLMAdjudicationConfig

class AppConfig(TypedDict):
    global_: _GlobalConfig   # accessed via cfg["global"] (key name in YAML is "global")
    figures: _FiguresConfig
    tables: _TablesConfig
    llm: _LLMConfig


# ── Required key validation ───────────────────────────────────────────────────

_REQUIRED_PATHS = [
    # (dotted path,)
    ("global.output_base_dir",),
    ("global.intermediate_dir",),
    ("figures.step2_macro.model_path",),
    ("figures.step3_micro.model_path",),
    ("tables.segmentation.model_path",),
    ("tables.structure.model_path",),
    # LLM/VLM configuration moved to typed Pydantic models loaded via
    # src.llm.config.load_llm_config / load_vlm_config; presence is
    # validated there.
    ("llm.adjudication.provider",),
    ("llm.adjudication.model",),
    ("vlm.inspection.provider",),
    ("vlm.inspection.model",),
]


def _check_required(cfg: dict) -> None:
    """Raise ValueError if any required config key is missing."""
    missing = []
    for (path,) in _REQUIRED_PATHS:
        parts = path.split(".")
        node = cfg
        for p in parts:
            if not isinstance(node, dict) or p not in node:
                missing.append(path)
                break
            node = node[p]
    if missing:
        raise ValueError(
            f"config.yaml is missing required key(s): {missing}. "
            "Fix config.yaml before starting the service."
        )


# ── Loader ────────────────────────────────────────────────────────────────────

_CONFIG_CACHE = None


def load_config(path: str = "config.yaml") -> dict:
    """Load and validate config.yaml. Raises on missing required keys.
    Caches the result so subsequent calls don't re-read the file.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    candidates = [
        path,
        os.path.join(os.getcwd(), path),
        os.path.join(os.path.dirname(__file__), "..", "..", path),
    ]

    config_path = None
    for c in candidates:
        if os.path.exists(c):
            config_path = c
            break

    if not config_path:
        raise FileNotFoundError(
            f"Configuration file '{path}' not found in search paths: {candidates}"
        )

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    _check_required(cfg)
    _CONFIG_CACHE = cfg
    return _CONFIG_CACHE
