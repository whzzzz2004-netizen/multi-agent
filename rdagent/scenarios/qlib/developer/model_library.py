"""Small persistent model library for Qlib research workflows."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from rdagent.log import rdagent_logger as logger

MODEL_LIBRARY_DIR = Path.cwd() / "git_ignore_folder" / "research_store" / "model_library"
MANIFEST_NAME = "manifest.json"

DEFAULT_MODELS: list[dict[str, Any]] = [
    {
        "name": "lgbm_default",
        "type": "qlib_builtin",
        "status": "stable",
        "priority": 10,
        "module_path": "qlib.contrib.model.gbdt",
        "class": "LGBModel",
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.05,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
        },
        "description": "Qlib built-in LightGBM baseline for factor-pool backtests.",
    },
    {
        "name": "lgbm_fast",
        "type": "qlib_builtin",
        "status": "stable",
        "priority": 20,
        "module_path": "qlib.contrib.model.gbdt",
        "class": "LGBModel",
        "kwargs": {
            "loss": "mse",
            "learning_rate": 0.08,
            "num_leaves": 64,
            "max_depth": 6,
            "num_threads": 20,
            "num_boost_round": 300,
            "early_stopping_rounds": 30,
        },
        "description": "Faster Qlib built-in LightGBM variant for quick iteration.",
    },
    {
        "name": "xgb_default",
        "type": "qlib_builtin",
        "status": "stable",
        "priority": 30,
        "module_path": "qlib.contrib.model.xgboost",
        "class": "XGBModel",
        "kwargs": {
            "objective": "reg:squarederror",
            "eta": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "nthread": 20,
        },
        "description": "Qlib built-in XGBoost baseline for factor-pool backtests.",
    },
]


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def _manifest_path(model_library_path: str | Path | None = None) -> Path:
    root = Path(model_library_path) if model_library_path is not None else MODEL_LIBRARY_DIR
    return root / MANIFEST_NAME


def ensure_model_library(model_library_path: str | Path | None = None) -> dict[str, Any]:
    """Create the model library and default built-in models if missing."""
    manifest_path = _manifest_path(model_library_path)
    root = manifest_path.parent
    root.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"version": 1, "created_at": datetime.now().isoformat(timespec="seconds"), "models": []}

    existing = {model["name"] for model in manifest.get("models", [])}
    changed = False
    for model in DEFAULT_MODELS:
        if model["name"] in existing:
            continue
        model_spec = deepcopy(model)
        model_spec["created_at"] = datetime.now().isoformat(timespec="seconds")
        model_spec["updated_at"] = model_spec["created_at"]
        model_spec["runs"] = []
        manifest.setdefault("models", []).append(model_spec)
        changed = True

        model_dir = root / model_spec["status"] / model_spec["name"]
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "metadata.json").write_text(
            json.dumps(model_spec, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )

    if changed or not manifest_path.exists():
        save_model_library(manifest, model_library_path=model_library_path)
    logger.info(f"Model library ready at {root}; models={len(manifest.get('models', []))}")
    return manifest


def save_model_library(manifest: dict[str, Any], model_library_path: str | Path | None = None) -> None:
    manifest_path = _manifest_path(model_library_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def load_model_library(model_library_path: str | Path | None = None) -> dict[str, Any]:
    return ensure_model_library(model_library_path=model_library_path)


def list_models(model_library_path: str | Path | None = None) -> list[dict[str, Any]]:
    manifest = load_model_library(model_library_path=model_library_path)
    return sorted(manifest.get("models", []), key=lambda m: (m.get("priority", 999), m.get("name", "")))


def get_model_spec(name: str, model_library_path: str | Path | None = None) -> dict[str, Any]:
    for model in list_models(model_library_path=model_library_path):
        if model.get("name") == name:
            return model
    raise KeyError(f"Model not found in model library: {name}")


def add_model_variant(
    base_model_name: str,
    variant_name: str,
    kwargs_patch: dict[str, Any],
    reason: str = "",
    model_library_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create or update a lightweight model-library variant by patching model kwargs."""
    manifest = load_model_library(model_library_path=model_library_path)
    root = _manifest_path(model_library_path).parent
    base_model = None
    for model in manifest.get("models", []):
        if model.get("name") == base_model_name:
            base_model = deepcopy(model)
            break
    if base_model is None:
        raise KeyError(f"Base model not found in model library: {base_model_name}")

    variant = deepcopy(base_model)
    variant["name"] = variant_name
    variant["status"] = "experimental"
    variant["priority"] = int(base_model.get("priority", 999)) + 100
    variant["description"] = f"LLM-tuned variant of {base_model_name}. {reason}".strip()
    variant["base_model"] = base_model_name
    variant["created_by"] = "llm_decision"
    variant["updated_at"] = datetime.now().isoformat(timespec="seconds")
    variant["created_at"] = variant.get("created_at", variant["updated_at"])
    patched_kwargs = deepcopy(base_model.get("kwargs", {}))
    patched_kwargs.update(kwargs_patch or {})
    variant["kwargs"] = patched_kwargs
    variant["runs"] = []
    variant.pop("last_backtest", None)

    models = manifest.setdefault("models", [])
    for idx, model in enumerate(models):
        if model.get("name") == variant_name:
            models[idx] = variant
            break
    else:
        models.append(variant)

    save_model_library(manifest, model_library_path=model_library_path)
    model_dir = root / variant["status"] / variant["name"]
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "metadata.json").write_text(
        json.dumps(variant, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    logger.info(f"Registered model-library variant {variant_name} from {base_model_name}: {kwargs_patch}")
    return variant


def register_custom_model(
    model_name: str,
    model_code: str,
    model_type: str = "custom_pytorch",
    description: str = "",
    architecture: str = "",
    formulation: str = "",
    variables: dict[str, Any] | None = None,
    hyperparameters: dict[str, Any] | None = None,
    training_hyperparameters: dict[str, Any] | None = None,
    source_paper_path: str | None = None,
    source_paper_title: str | None = None,
    model_library_path: str | Path | None = None,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Register an implemented literature/custom model into the persistent model library."""
    manifest = load_model_library(model_library_path=model_library_path)
    root = _manifest_path(model_library_path).parent
    model_spec = {
        "name": model_name,
        "type": model_type,
        "status": "experimental",
        "priority": 200,
        "description": description,
        "architecture": architecture,
        "formulation": formulation,
        "variables": variables or {},
        "hyperparameters": hyperparameters or {},
        "training_hyperparameters": training_hyperparameters or {},
        "source_paper_path": source_paper_path,
        "source_paper_title": source_paper_title,
        "created_by": "paper_import",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "runs": [],
    }

    models = manifest.setdefault("models", [])
    existing_idx = None
    for idx, model in enumerate(models):
        if model.get("name") == model_name:
            existing_idx = idx
            break

    if existing_idx is not None and not overwrite:
        raise ValueError(f"Model already exists in model library: {model_name}")

    if existing_idx is not None:
        old_model = models[existing_idx]
        if old_model.get("runs"):
            model_spec["runs"] = old_model["runs"]
        if old_model.get("last_backtest"):
            model_spec["last_backtest"] = old_model["last_backtest"]
        model_spec["created_at"] = old_model.get("created_at", model_spec["created_at"])
        models[existing_idx] = model_spec
    else:
        models.append(model_spec)

    save_model_library(manifest, model_library_path=model_library_path)

    model_dir = root / model_spec["status"] / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.py").write_text(model_code, encoding="utf-8")
    (model_dir / "metadata.json").write_text(
        json.dumps(model_spec, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    logger.info(f"Registered custom model in library: {model_name}; path={model_dir}")
    return model_spec


def select_model(model_name: str | None = None, model_library_path: str | Path | None = None) -> dict[str, Any]:
    """Select a model. Prefer explicit name, then untried stable models, then best known model."""
    if model_name:
        return get_model_spec(model_name, model_library_path=model_library_path)

    models = [m for m in list_models(model_library_path=model_library_path) if m.get("status") != "disabled"]
    untried = [m for m in models if not m.get("runs")]
    if untried:
        selected = sorted(untried, key=lambda m: (m.get("priority", 999), m.get("name", "")))[0]
        logger.info(f"Auto-selected untried model from library: {selected['name']}")
        return selected

    def score(model: dict[str, Any]) -> float:
        metrics = (model.get("runs") or [{}])[-1].get("metrics") or {}
        for key in [
            "1day.excess_return_with_cost.information_ratio",
            "1day.excess_return_with_cost.annualized_return",
            "1day.excess_return_without_cost.information_ratio",
        ]:
            val = metrics.get(key)
            if isinstance(val, (int, float)):
                return float(val)
        return float("-inf")

    selected = sorted(models, key=lambda m: (score(m), -m.get("priority", 999)), reverse=True)[0]
    logger.info(f"Auto-selected best recorded model from library: {selected['name']}")
    return selected


def record_model_run(
    model_name: str,
    metrics: dict[str, Any],
    workspace_path: str,
    factor_top_k: int | None = None,
    factor_offset: int = 0,
    factor_names: list[str] | None = None,
    date_split: dict[str, str] | None = None,
    model_library_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest = load_model_library(model_library_path=model_library_path)
    root = _manifest_path(model_library_path).parent
    for model in manifest.get("models", []):
        if model.get("name") != model_name:
            continue
        run = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "workspace_path": workspace_path,
            "factor_top_k": factor_top_k,
            "factor_offset": factor_offset,
            "factor_names": factor_names or [],
            "date_split": date_split or {},
            "metrics": metrics,
        }
        model.setdefault("runs", []).append(run)
        model["last_backtest"] = run
        model["updated_at"] = run["created_at"]
        save_model_library(manifest, model_library_path=model_library_path)
        model_dir = root / model.get("status", "experimental") / model["name"]
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "metadata.json").write_text(
            json.dumps(model, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        logger.info(f"Recorded model-library run for {model_name}: workspace={workspace_path}")
        return model
    raise KeyError(f"Model not found in model library: {model_name}")
