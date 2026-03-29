import os
import sys
from pathlib import Path


def configure_runtime_env() -> Path:
    """Redirect third-party cache/config writes into the project workspace."""
    project_root = Path(__file__).resolve().parents[2]
    original_home = Path(
        os.environ.get("FLOWFIGTABMINER_ORIGINAL_HOME")
        or os.environ.get("USERPROFILE")
        or os.path.expanduser("~")
    )
    runtime_root = project_root / ".runtime_cache"
    home_dir = runtime_root / "home"
    xdg_cache = runtime_root / "xdg_cache"
    xdg_config = runtime_root / "xdg_config"
    yolo_dir = runtime_root / "yolo"
    paddle_dir = runtime_root / "paddle"
    paddlex_dir = runtime_root / "paddlex"
    paddleocr_dir = runtime_root / "paddleocr"
    models_hub_dir = project_root / "models" / "hub"
    hf_dir = models_hub_dir if models_hub_dir.exists() else runtime_root / "huggingface"
    easyocr_dir = original_home / ".EasyOCR" if (original_home / ".EasyOCR").exists() else runtime_root / "EasyOCR"

    for path in [runtime_root, home_dir, xdg_cache, xdg_config, yolo_dir, paddle_dir, paddlex_dir, paddleocr_dir, hf_dir, easyocr_dir]:
        path.mkdir(parents=True, exist_ok=True)

    os.environ["FLOWFIGTABMINER_ORIGINAL_HOME"] = str(original_home)
    os.environ["HOME"] = str(home_dir)
    os.environ["USERPROFILE"] = str(home_dir)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache)
    os.environ["XDG_CONFIG_HOME"] = str(xdg_config)
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_dir)
    os.environ["PADDLE_HOME"] = str(paddle_dir)
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(paddlex_dir)
    os.environ["PADDLEOCR_HOME"] = str(paddleocr_dir)
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["HF_HOME"] = str(hf_dir)
    os.environ["EASYOCR_MODULE_PATH"] = str(easyocr_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_dir / "transformers")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_dir / "hub")

    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        dll_dirs = [
            Path(sys.prefix) / "Library" / "bin",
            Path(sys.prefix) / "Scripts",
            Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib",
        ]
        for dll_dir in dll_dirs:
            if dll_dir.exists():
                os.add_dll_directory(str(dll_dir))

    return runtime_root
