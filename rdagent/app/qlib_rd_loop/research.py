"""
Research-mode workflows for Qlib.

These entry points keep the original fin_factor/fin_model/fin_quant flows intact and
provide narrower workflows for day-to-day research operations.
"""

import asyncio
import json
import math
import re
from pathlib import Path
from typing import Any, Optional

import fire

from rdagent.components.coder.model_coder.task_loader import ModelExperimentLoaderFromPDFfiles
from rdagent.components.document_reader.document_reader import extract_first_page_screenshot_from_pdf
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.components.coder.factor_coder.eva_utils import evaluate_factor_ic_from_workspace
from rdagent.app.qlib_rd_loop.conf import FACTOR_PROP_SETTING, MODEL_PROP_SETTING
from rdagent.app.qlib_rd_loop.factor import FactorRDLoop
from rdagent.app.qlib_rd_loop.model import ModelRDLoop
from rdagent.core.exception import ModelEmptyError
from rdagent.core.proposal import HypothesisFeedback
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.general_model.scenario import GeneralModelScenario
from rdagent.scenarios.qlib.developer.factor_pool import load_factor_pool
from rdagent.scenarios.qlib.developer.model_library import (
    add_model_variant,
    ensure_model_library,
    list_models,
    record_model_run,
    register_custom_model,
    select_model,
)
from rdagent.scenarios.qlib.developer.model_coder import QlibModelCoSTEER
from rdagent.scenarios.qlib.developer.model_runner import QlibModelRunner
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment
from rdagent.scenarios.qlib.experiment.workspace import QlibFBWorkspace


class FactorMiningLoop(FactorRDLoop):
    """Mine factors into the local factor pool without running Qlib backtests."""

    skip_loop_error = ()
    skip_loop_error_stepname = None

    def __init__(self, *args, batch_size: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.mined_factor_count = 0

    async def direct_exp_gen(self, prev_out):
        if self.mined_factor_count >= self.batch_size:
            raise self.LoopTerminationError(f"Factor mining batch limit reached: {self.batch_size}")

        result = await super().direct_exp_gen(prev_out)
        exp = result["exp_gen"]
        remaining = self.batch_size - self.mined_factor_count
        if len(exp.sub_tasks) > remaining:
            logger.info(
                f"Factor mining batch has {remaining} slots left; truncating generated factors from "
                f"{len(exp.sub_tasks)} to {remaining}."
            )
            exp.sub_tasks = exp.sub_tasks[:remaining]
            exp.sub_workspace_list = exp.sub_workspace_list[:remaining]
        return result

    def record(self, prev_out):
        exp = prev_out.get("coding") or prev_out.get("direct_exp_gen", {}).get("exp_gen")
        exported_count = 0
        if exp is not None and getattr(exp, "prop_dev_feedback", None) is not None:
            for task, workspace, feedback in zip(exp.sub_tasks, exp.sub_workspace_list, exp.prop_dev_feedback):
                if workspace is None or feedback is None or not bool(feedback.final_decision):
                    continue
                try:
                    _, df = workspace.execute("All")
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Failed to reload factor dataframe for reviewed export: task={task.factor_name}; {exc}")
                    continue
                if df is None or df.empty:
                    continue
                ic_feedback, full_sample_ic = evaluate_factor_ic_from_workspace(workspace, data_type="All")
                if full_sample_ic is None or abs(full_sample_ic) < FACTOR_COSTEER_SETTINGS.min_abs_ic:
                    logger.info(
                        f"Skip reviewed export for factor {task.factor_name} because full-sample IC did not pass. "
                        f"{ic_feedback}"
                    )
                    continue
                logic_summary = task.factor_description
                review_notes = "\n".join(
                    part for part in [feedback.execution, feedback.return_checking, feedback.code, ic_feedback] if part
                )
                tags = _infer_factor_registry_tags(task, feedback)
                tags.append("ic_passed")
                tags = sorted(set(tags))
                workspace.export_reviewed_factor(
                    df,
                    accepted=True,
                    logic_summary=logic_summary,
                    tags=tags,
                    review_notes=review_notes,
                    ic_score=full_sample_ic,
                )
                exported_count += 1
        feedback = HypothesisFeedback(
            reason=(
                "Factor mining mode finished after coding/evaluation. "
                "Generated factor files are exported to git_ignore_folder/factor_outputs; "
                "Qlib backtest was intentionally skipped."
            ),
            decision=True,
            code_change_summary="Mined factors into the factor pool without running a backtest.",
            acceptable=True,
        )
        logger.log_object(feedback, tag="factor mining feedback")
        if exp is not None:
            self.mined_factor_count += len(exp.sub_tasks)
            self.trace.sync_dag_parent_and_hist((exp, feedback), prev_out[self.LOOP_IDX_KEY])
        logger.info(
            f"Factor mining loop recorded. Backtest step was skipped by design. "
            f"Batch progress: {self.mined_factor_count}/{self.batch_size}. "
            f"Accepted factor exports: {exported_count}."
        )


def _infer_factor_registry_tags(task, feedback) -> list[str]:
    content = " ".join(
        [
            getattr(task, "factor_name", "") or "",
            getattr(task, "factor_description", "") or "",
            getattr(task, "factor_formulation", "") or "",
            str(getattr(task, "variables", {}) or {}),
            getattr(feedback, "code", "") or "",
        ]
    ).lower()
    tags: set[str] = set()
    keyword_to_tag = {
        "momentum": "momentum",
        "reversal": "reversal",
        "volatility": "volatility",
        "volume": "volume",
        "vwap": "vwap",
        "spread": "liquidity",
        "liquidity": "liquidity",
        "bid": "quote",
        "ask": "quote",
        "minute": "minute_input",
        "intraday": "minute_input",
        "microstructure": "microstructure",
        "gap": "gap",
        "trend": "trend",
        "acceleration": "acceleration",
    }
    for keyword, tag in keyword_to_tag.items():
        if keyword in content:
            tags.add(tag)
    if "future-information leak" not in content and "time leakage" not in content:
        tags.add("leakage_checked")
    return sorted(tags)


# Keep only propose -> coding -> record. The inherited FactorRDLoop still exists unchanged;
# this class narrows the workflow for the factor-pool mining mode only.
FactorMiningLoop.steps = ["direct_exp_gen", "coding", "record"]


class QlibModelFromFactorPoolRunner(QlibModelRunner):
    """Run generated models directly on the local factor pool without a baseline run."""

    def __init__(self, *args, factor_pool_path: str | Path | None = None, factor_top_k: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.factor_pool_path = factor_pool_path
        self.factor_top_k = factor_top_k

    def develop(self, exp: QlibModelExperiment) -> QlibModelExperiment:
        if exp.sub_workspace_list[0].file_dict.get("model.py") is None:
            raise ModelEmptyError("model.py is empty")

        combined_factors = load_factor_pool(self.factor_pool_path, self.factor_top_k)
        target_path = exp.experiment_workspace.workspace_path / "combined_factors_df.parquet"
        combined_factors.to_parquet(target_path, engine="pyarrow")
        logger.info(f"Factor pool dataframe saved to {target_path}")

        exp.experiment_workspace.inject_files(**{"model.py": exp.sub_workspace_list[0].file_dict["model.py"]})

        env_to_use = {
            "PYTHONPATH": "./",
            "num_features": str(len(combined_factors.columns)),
        }
        env_to_use.update(_factor_pool_date_env(combined_factors))

        training_hyperparameters = exp.sub_tasks[0].training_hyperparameters
        env_to_use.update(
            {
                "n_epochs": str((training_hyperparameters or {}).get("n_epochs", "100")),
                "lr": str((training_hyperparameters or {}).get("lr", "2e-4")),
                "early_stop": str((training_hyperparameters or {}).get("early_stop", 10)),
                "batch_size": str((training_hyperparameters or {}).get("batch_size", 256)),
                "weight_decay": str((training_hyperparameters or {}).get("weight_decay", 0.0001)),
            }
        )

        logger.info(
            f"Running model {exp.sub_tasks[0].name} on factor pool with {len(combined_factors.columns)} factors."
        )
        if exp.sub_tasks[0].model_type == "TimeSeries":
            env_to_use.update({"dataset_cls": "TSDatasetH", "step_len": 20, "num_timesteps": 20})
        else:
            env_to_use.update({"dataset_cls": "DatasetH"})

        result, stdout = exp.experiment_workspace.execute(
            qlib_config_name="conf_pool_factors_model.yaml",
            run_env=env_to_use,
        )
        exp.result = result
        exp.stdout = stdout
        if result is None:
            logger.error(f"Failed to run {exp.sub_tasks[0].name} on factor pool, because {stdout}")
            raise ModelEmptyError(f"Failed to run {exp.sub_tasks[0].name} on factor pool, because {stdout}")
        return exp


class ModelFromPoolLoop(ModelRDLoop):
    """Generate models and backtest them directly on the existing factor pool."""

    def __init__(
        self,
        factor_pool_path: str | Path | None = None,
        factor_top_k: int | None = None,
    ):
        super().__init__(MODEL_PROP_SETTING)
        self.plan["features"] = {}
        self.plan["feature_codes"] = {}
        self.runner = QlibModelFromFactorPoolRunner(
            self.trace.scen,
            factor_pool_path=factor_pool_path,
            factor_top_k=factor_top_k,
        )
        logger.info("Model-from-pool mode initialized: baseline features and baseline selection are disabled.")


def _factor_pool_date_env(combined_factors) -> dict[str, str]:
    """Use the actual factor-pool calendar instead of the broad default dates."""
    dates = combined_factors.index.get_level_values("datetime").unique().sort_values()
    if len(dates) < 10:
        raise ValueError(f"Factor pool has too few trading dates to split train/valid/test: {len(dates)}")

    train_end_idx = max(1, int(len(dates) * 0.6) - 1)
    valid_start_idx = min(train_end_idx + 1, len(dates) - 3)
    valid_end_idx = max(valid_start_idx, int(len(dates) * 0.8) - 1)
    test_start_idx = min(valid_end_idx + 1, len(dates) - 2)

    def fmt(idx: int) -> str:
        return dates[idx].strftime("%Y-%m-%d")

    return {
        "train_start": fmt(0),
        "train_end": fmt(train_end_idx),
        "valid_start": fmt(valid_start_idx),
        "valid_end": fmt(valid_end_idx),
        "test_start": fmt(test_start_idx),
        "test_end": fmt(len(dates) - 1),
    }


def _factor_names_from_df(combined_factors) -> list[str]:
    cols = combined_factors.columns
    if hasattr(cols, "get_level_values") and getattr(cols, "nlevels", 1) > 1:
        return [str(col) for col in cols.get_level_values(-1)]
    return [str(col) for col in cols]


def _metric_score(metrics: dict) -> float:
    for key in [
        "1day.excess_return_with_cost.information_ratio",
        "1day.excess_return_with_cost.annualized_return",
        "1day.excess_return_without_cost.information_ratio",
    ]:
        val = metrics.get(key)
        if isinstance(val, (int, float)) and not math.isnan(float(val)):
            return float(val)
    return float("-inf")


def _metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "1day.excess_return_with_cost.information_ratio",
        "1day.excess_return_with_cost.annualized_return",
        "1day.excess_return_with_cost.max_drawdown",
        "1day.excess_return_without_cost.information_ratio",
        "IC",
        "ICIR",
        "Rank IC",
        "Rank ICIR",
        "l2.train",
        "l2.valid",
    ]
    return {key: metrics.get(key) for key in keys if key in metrics}


def _factor_slice_preview(
    factor_pool_path: str | None,
    top_k: int,
    start_offset: int = 0,
    slice_count: int = 5,
) -> list[dict[str, Any]]:
    previews = []
    for idx in range(slice_count):
        offset = start_offset + idx * top_k
        try:
            df = load_factor_pool(factor_pool_path, top_k, factor_offset=offset)
            previews.append(
                {
                    "factor_offset": offset,
                    "factor_top_k": top_k,
                    "factor_count": len(df.columns),
                    "factor_names": _factor_names_from_df(df),
                }
            )
        except Exception as e:  # noqa: BLE001 - preview is advisory; the run path will raise real errors.
            previews.append({"factor_offset": offset, "factor_top_k": top_k, "error": str(e)})
            break
    return previews


def _parse_llm_json_response(response: str) -> dict[str, Any]:
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        start = response.find("{")
        end = response.rfind("}")
        if start >= 0 and end > start:
            return json.loads(response[start : end + 1])
        raise


def _llm_decide_next_backtest_round(
    history: list[dict[str, Any]],
    candidate_models: list[dict[str, Any]],
    current_model: dict[str, Any],
    factor_pool_path: str | None,
    top_k: int,
    min_cost_ir: float,
) -> dict[str, Any]:
    """Ask the chat model whether the next round should switch factors, switch model, or tune params."""
    model_brief = [
        {
            "name": model.get("name"),
            "type": model.get("type"),
            "status": model.get("status"),
            "class": model.get("class"),
            "module_path": model.get("module_path"),
            "kwargs": model.get("kwargs", {}),
        }
        for model in candidate_models
        if model.get("type") == "qlib_builtin"
    ]
    last_offset = int(history[-1].get("factor_offset", 0)) if history else 0
    factor_previews = _factor_slice_preview(
        factor_pool_path=factor_pool_path,
        top_k=top_k,
        start_offset=last_offset + top_k,
        slice_count=5,
    )
    prompt = {
        "goal": (
            "Decide the next Qlib backtest round for a quant research workflow. "
            "The user wants the LLM to choose whether to change factor slice, switch model, "
            "or tune model hyperparameters after seeing the latest backtest result."
        ),
        "target": {"min_cost_adjusted_information_ratio": min_cost_ir},
        "current_model": {
            "name": current_model.get("name"),
            "class": current_model.get("class"),
            "module_path": current_model.get("module_path"),
            "kwargs": current_model.get("kwargs", {}),
        },
        "candidate_models": model_brief,
        "recent_history": history[-5:],
        "available_next_factor_slices": factor_previews,
        "decision_schema": {
            "action": "one of: change_factors, switch_model, tune_model, stop",
            "model_name": "existing model name when action is switch_model; otherwise optional",
            "factor_offset": "integer offset into factor pool; use one shown in available_next_factor_slices",
            "factor_top_k": "integer number of factors to use; usually keep current top_k",
            "model_kwargs_patch": "small dict of qlib model kwargs to update when action is tune_model",
            "reason": "short Chinese explanation for logging",
        },
        "constraints": [
            "Prefer changing factor slice when the model trains but IC/RankIC are NaN or backtest IR is poor.",
            "Prefer tuning LightGBM only with conservative kwargs patches such as learning_rate, num_leaves, max_depth, lambda_l1, lambda_l2, subsample, colsample_bytree.",
            "Do not invent a model not listed in candidate_models.",
            "Return only one JSON object. No markdown.",
        ],
    }
    response = APIBackend(use_chat_cache=False).build_messages_and_create_chat_completion(
        user_prompt=json.dumps(prompt, ensure_ascii=False, indent=2),
        system_prompt=(
            "You are a careful quant research controller. Decide one next experiment, "
            "balancing signal search and runtime stability. Return strict JSON only."
        ),
        json_mode=True,
        json_target_type=dict,
    )
    decision = _parse_llm_json_response(response)
    action = str(decision.get("action", "change_factors"))
    if action not in {"change_factors", "switch_model", "tune_model", "stop"}:
        action = "change_factors"
    decision["action"] = action
    decision["factor_top_k"] = int(decision.get("factor_top_k") or top_k)
    decision["factor_offset"] = int(decision.get("factor_offset") or (last_offset + top_k))
    if not isinstance(decision.get("model_kwargs_patch"), dict):
        decision["model_kwargs_patch"] = {}
    logger.info(f"LLM backtest decision: {json.dumps(decision, ensure_ascii=False)}")
    return decision


def mine_factors(
    path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    all_duration: str | None = None,
    checkout: bool = True,
    checkout_path: Optional[str] = None,
    base_features_path: Optional[str] = None,
    batch_size: int = 10,
    **kwargs,
):
    """Mine new factors and persist them to git_ignore_folder/factor_outputs without Qlib backtesting."""
    if checkout_path is not None:
        checkout = Path(checkout_path)

    if path is None:
        factor_loop = FactorMiningLoop(FACTOR_PROP_SETTING, batch_size=batch_size)
    else:
        factor_loop = FactorMiningLoop.load(path, checkout=checkout)
        factor_loop.batch_size = batch_size

    factor_loop._init_base_features(base_features_path)
    if "user_interaction_queues" in kwargs and kwargs["user_interaction_queues"] is not None:
        factor_loop._set_interactor(*kwargs["user_interaction_queues"])
        factor_loop._interact_init_params()

    logger.info(f"Starting factor mining mode: Qlib backtest will be skipped; batch_size={batch_size}.")
    asyncio.run(factor_loop.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))


def model_from_pool(
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    all_duration: str | None = None,
    factor_pool_path: str | None = None,
    factor_top_k: int | None = None,
    **kwargs,
):
    """Try generated models on the existing factor pool and run Qlib backtests without baseline selection."""
    model_loop = ModelFromPoolLoop(factor_pool_path=factor_pool_path, factor_top_k=factor_top_k)
    if "user_interaction_queues" in kwargs and kwargs["user_interaction_queues"] is not None:
        model_loop._set_interactor(*kwargs["user_interaction_queues"])
        model_loop._interact_init_params()

    logger.info("Starting model-from-pool mode: using existing factor pool; baseline run is skipped.")
    asyncio.run(model_loop.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))


def lgbm_from_pool(
    factor_pool_path: str | None = None,
    factor_top_k: int | None = None,
):
    """Run a Qlib built-in LightGBM backtest directly on the existing factor pool.

    This intentionally skips model proposal, custom PyTorch model generation, and CoSTEER
    model-code evaluation. It is the low-noise path for quickly checking whether the
    current factor pool has backtest value.
    """
    ensure_model_library()
    model_spec = select_model("lgbm_default")
    return _run_builtin_model_from_pool(
        model_spec=model_spec,
        factor_pool_path=factor_pool_path,
        factor_top_k=factor_top_k,
        record_to_library=True,
    )


def _run_builtin_model_from_pool(
    model_spec: dict,
    factor_pool_path: str | None = None,
    factor_top_k: int | None = None,
    factor_offset: int = 0,
    model_library_path: str | None = None,
    record_to_library: bool = True,
):
    if model_spec.get("type") != "qlib_builtin":
        raise ValueError(f"Only qlib_builtin models are supported in this short path, got: {model_spec.get('type')}")

    model_template_path = Path(__file__).resolve().parents[2] / "scenarios" / "qlib" / "experiment" / "model_template"
    workspace = QlibFBWorkspace(template_folder_path=model_template_path)

    combined_factors = load_factor_pool(factor_pool_path, factor_top_k, factor_offset=factor_offset)
    factor_names = _factor_names_from_df(combined_factors)
    target_path = workspace.workspace_path / "combined_factors_df.parquet"
    combined_factors.to_parquet(target_path, engine="pyarrow")
    factor_selection_path = workspace.workspace_path / "factor_selection.json"
    factor_selection_path.write_text(
        json.dumps(
            {
                "factor_top_k": factor_top_k,
                "factor_offset": factor_offset,
                "factor_count": len(factor_names),
                "factor_names": factor_names,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info(
        f"Built-in model mode saved factor pool dataframe to {target_path}; "
        f"shape={combined_factors.shape}; factor_selection={factor_selection_path}."
    )
    logger.info(f"Using factors: {factor_names}")

    date_split = _factor_pool_date_env(combined_factors)
    env_to_use = {
        "PYTHONPATH": "./",
        "model_class": model_spec["class"],
        "model_module_path": model_spec["module_path"],
        "model_kwargs": json.dumps(model_spec.get("kwargs", {}), ensure_ascii=False),
    }
    env_to_use.update(date_split)
    logger.info(f"Built-in model mode uses factor-pool date split: {env_to_use}")

    logger.info(f"Starting built-in Qlib model {model_spec['name']} on factor pool; no custom model.py will be generated.")
    result, stdout = workspace.execute(
        qlib_config_name="conf_pool_factors_builtin.yaml",
        run_env=env_to_use,
    )
    if result is None:
        raise RuntimeError(f"Built-in model factor-pool backtest failed for {model_spec['name']}. {stdout}")

    metrics = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    if record_to_library:
        record_model_run(
            model_spec["name"],
            metrics=metrics,
            workspace_path=str(workspace.workspace_path),
            factor_top_k=factor_top_k,
            factor_offset=factor_offset,
            factor_names=factor_names,
            date_split=date_split,
            model_library_path=model_library_path,
        )

    logger.info(f"Built-in model factor-pool backtest finished for {model_spec['name']}. Metrics:\n{result}")
    return result


def list_model_library(model_library_path: str | None = None):
    """Print available models in the persistent model library."""
    models = list_models(model_library_path=model_library_path)
    for model in models:
        runs = len(model.get("runs") or [])
        last = (model.get("last_backtest") or {}).get("metrics") or {}
        score = last.get("1day.excess_return_with_cost.information_ratio", "NA")
        logger.info(
            f"model={model['name']} type={model.get('type')} status={model.get('status')} "
            f"runs={runs} last_cost_ir={score}"
        )
    return models


def import_models_from_report(
    report_file_path: str,
    model_library_path: str | None = None,
):
    """Read a paper/report PDF, implement extracted models, and save them into the model library."""
    ensure_model_library(model_library_path=model_library_path)
    paper_path = str(Path(report_file_path).resolve())
    logger.info(f"Importing literature models from report: {paper_path}")

    try:
        img = extract_first_page_screenshot_from_pdf(report_file_path)
        logger.log_object(img, tag="paper_first_page")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to extract first-page screenshot from {report_file_path}: {e}")

    exp = ModelExperimentLoaderFromPDFfiles().load(report_file_path)
    logger.log_object(exp, tag="paper_loaded_model_experiment")
    exp = QlibModelCoSTEER(GeneralModelScenario()).develop(exp)
    logger.log_object(exp, tag="paper_developed_model_experiment")

    imported_models = []
    for task, workspace in zip(exp.sub_tasks, exp.sub_workspace_list):
        if workspace is None or workspace.file_dict.get("model.py") is None:
            logger.warning(f"Skip model {task.name}: no implemented model.py found after paper import.")
            continue
        model_code = workspace.file_dict["model.py"]
        model_spec = register_custom_model(
            model_name=task.name,
            model_code=model_code,
            model_type="custom_pytorch",
            description=task.description,
            architecture=task.architecture,
            formulation=task.formulation,
            variables=task.variables,
            hyperparameters=task.hyperparameters,
            training_hyperparameters=task.training_hyperparameters,
            source_paper_path=paper_path,
            source_paper_title=Path(report_file_path).stem,
            model_library_path=model_library_path,
            overwrite=True,
        )
        imported_models.append(model_spec)

    logger.info(
        f"Imported {len(imported_models)} literature models into model library from {paper_path}."
    )
    return imported_models


def backtest_model_library(
    model_name: str | None = None,
    factor_pool_path: str | None = None,
    factor_top_k: int | None = None,
    model_library_path: str | None = None,
    auto_rounds: int = 10,
    min_cost_ir: float = 0.0,
    llm_decision: bool = True,
):
    """Auto-pick a model from the model library and backtest it on the factor pool.

    If auto_rounds > 1 and the cost-adjusted information ratio is below min_cost_ir,
    the chat model decides whether to change the factor slice, switch model, or create
    a conservative tuned variant in the model library.
    """
    ensure_model_library(model_library_path=model_library_path)
    if auto_rounds < 1:
        auto_rounds = 1

    if model_name:
        candidate_models = [select_model(model_name=model_name, model_library_path=model_library_path)]
    else:
        candidate_models = [m for m in list_models(model_library_path=model_library_path) if m.get("status") != "disabled"]

    best_result = None
    best_score = float("-inf")
    top_k = factor_top_k or 10
    tried = 0
    history: list[dict[str, Any]] = []
    current_model = candidate_models[0]
    current_factor_offset = 0
    for round_idx in range(auto_rounds):
        if round_idx == 0:
            model_spec = current_model
            factor_offset = current_factor_offset
        elif llm_decision:
            decision = _llm_decide_next_backtest_round(
                history=history,
                candidate_models=candidate_models,
                current_model=current_model,
                factor_pool_path=factor_pool_path,
                top_k=top_k,
                min_cost_ir=min_cost_ir,
            )
            if decision["action"] == "stop":
                logger.info(f"LLM decided to stop auto-search: {decision.get('reason', '')}")
                break
            factor_offset = int(decision.get("factor_offset", current_factor_offset + top_k))
            top_k = int(decision.get("factor_top_k") or top_k)
            if decision["action"] == "switch_model":
                requested_name = decision.get("model_name")
                matching = [m for m in candidate_models if m.get("name") == requested_name]
                if matching:
                    model_spec = matching[0]
                else:
                    logger.warning(
                        f"LLM requested unknown model {requested_name}; falling back to next candidate model."
                    )
                    model_spec = candidate_models[round_idx % len(candidate_models)]
            elif decision["action"] == "tune_model":
                patch = decision.get("model_kwargs_patch") or {}
                if patch:
                    variant_name = f"{current_model['name']}_llm_r{round_idx + 1}"
                    model_spec = add_model_variant(
                        base_model_name=current_model["name"],
                        variant_name=variant_name,
                        kwargs_patch=patch,
                        reason=str(decision.get("reason", "")),
                        model_library_path=model_library_path,
                    )
                    candidate_models.append(model_spec)
                else:
                    logger.info("LLM chose tune_model without a kwargs patch; changing factor slice instead.")
                    model_spec = current_model
            else:
                model_spec = current_model
        else:
            model_spec = candidate_models[round_idx % len(candidate_models)]
            factor_offset = round_idx * top_k

        if model_spec.get("type") == "custom_pytorch":
            logger.warning(
                f"Skip custom_pytorch model {model_spec.get('name')} in low-noise auto-search. "
                "Use fin_model_from_pool for generated PyTorch models until the custom registry is hardened."
            )
            continue
        if model_spec.get("type") != "qlib_builtin":
            logger.warning(f"Skip unsupported model type: {model_spec.get('type')}")
            continue

        logger.info(
            f"Auto-search round {round_idx + 1}/{auto_rounds}: "
            f"model={model_spec['name']} factor_top_k={top_k} factor_offset={factor_offset}"
        )
        result = _run_builtin_model_from_pool(
            model_spec=model_spec,
            factor_pool_path=factor_pool_path,
            factor_top_k=top_k,
            factor_offset=factor_offset,
            model_library_path=model_library_path,
            record_to_library=True,
        )
        tried += 1
        metrics = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        score = _metric_score(metrics)
        if score > best_score:
            best_score = score
            best_result = result
        if score >= min_cost_ir:
            logger.info(f"Auto-search reached target score {score} >= {min_cost_ir}; stopping.")
            return result
        current_model = model_spec
        current_factor_offset = factor_offset
        history.append(
            {
                "round": round_idx + 1,
                "model_name": model_spec.get("name"),
                "model_class": model_spec.get("class"),
                "factor_top_k": top_k,
                "factor_offset": factor_offset,
                "factor_names": _factor_names_from_df(load_factor_pool(factor_pool_path, top_k, factor_offset)),
                "score": score,
                "metrics": _metric_summary(metrics),
            }
        )

    logger.info(
        f"Auto-search finished after {tried} runnable rounds. "
        f"Best cost-adjusted score={best_score}; target={min_cost_ir}."
    )
    return best_result


def main(
    mode: str = "mine_factors",
    path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    all_duration: str | None = None,
    checkout: bool = True,
    checkout_path: Optional[str] = None,
    base_features_path: Optional[str] = None,
    factor_pool_path: Optional[str] = None,
    factor_top_k: Optional[int] = None,
    model_name: Optional[str] = None,
    model_library_path: Optional[str] = None,
    report_file_path: Optional[str] = None,
    auto_rounds: int = 10,
    min_cost_ir: float = 0.0,
    llm_decision: bool = True,
    batch_size: int = 10,
    **kwargs,
):
    """Research-mode dispatcher. Supports: mine_factors, model_from_pool, lgbm_from_pool."""
    if mode == "mine_factors":
        return mine_factors(
            path=path,
            step_n=step_n,
            loop_n=loop_n,
            all_duration=all_duration,
            checkout=checkout,
            checkout_path=checkout_path,
            base_features_path=base_features_path,
            batch_size=batch_size,
            **kwargs,
        )
    if mode == "model_from_pool":
        return model_from_pool(
            step_n=step_n,
            loop_n=loop_n,
            all_duration=all_duration,
            factor_pool_path=factor_pool_path,
            factor_top_k=factor_top_k,
            **kwargs,
        )
    if mode == "lgbm_from_pool":
        return lgbm_from_pool(
            factor_pool_path=factor_pool_path,
            factor_top_k=factor_top_k,
        )
    if mode == "model_library":
        return backtest_model_library(
            model_name=model_name,
            factor_pool_path=factor_pool_path,
            factor_top_k=factor_top_k,
            model_library_path=model_library_path,
            auto_rounds=auto_rounds,
            min_cost_ir=min_cost_ir,
            llm_decision=llm_decision,
        )
    if mode == "paper_to_model_library":
        if not report_file_path:
            raise ValueError("report_file_path is required when mode=paper_to_model_library")
        return import_models_from_report(
            report_file_path=report_file_path,
            model_library_path=model_library_path,
        )
    if mode == "list_model_library":
        return list_model_library(model_library_path=model_library_path)
    raise ValueError(
        "Unsupported research mode: "
        f"{mode}. Supported modes: mine_factors, model_from_pool, lgbm_from_pool, model_library, "
        "paper_to_model_library, list_model_library"
    )


if __name__ == "__main__":
    fire.Fire(main)
