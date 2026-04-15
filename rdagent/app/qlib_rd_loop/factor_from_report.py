import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import fire

from rdagent.app.qlib_rd_loop.conf import FACTOR_FROM_REPORT_PROP_SETTING
from rdagent.app.qlib_rd_loop.factor import FactorRDLoop
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.components.coder.factor_coder.eva_utils import evaluate_factor_ic_from_workspace
from rdagent.components.document_reader.document_reader import (
    extract_first_page_screenshot_from_pdf,
    load_and_process_pdfs_by_langchain,
)
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.proposal import Hypothesis, HypothesisFeedback
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.factor_experiment_loader.pdf_loader import (
    FactorExperimentLoaderFromPDFfiles,
)
from rdagent.utils.agent.tpl import T
from rdagent.utils.workflow import LoopMeta


def generate_hypothesis(factor_result: dict, report_content: str) -> str:
    """
    Generate a hypothesis based on factor results and report content.

    Args:
        factor_result (dict): The results of the factor analysis.
        report_content (str): The content of the report.

    Returns:
        str: The generated hypothesis.
    """
    system_prompt = T(".prompts:hypothesis_generation.system").r()
    user_prompt = T(".prompts:hypothesis_generation.user").r(
        factor_descriptions=json.dumps(factor_result), report_content=report_content
    )

    response = APIBackend().build_messages_and_create_chat_completion(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        json_mode=True,
        json_target_type=Dict[str, str],
    )

    response_json = json.loads(response)

    return Hypothesis(
        hypothesis=response_json.get("hypothesis", "No hypothesis provided"),
        reason=response_json.get("reason", "No reason provided"),
        concise_reason=response_json.get("concise_reason", "No concise reason provided"),
        concise_observation=response_json.get("concise_observation", "No concise observation provided"),
        concise_justification=response_json.get("concise_justification", "No concise justification provided"),
        concise_knowledge=response_json.get("concise_knowledge", "No concise knowledge provided"),
    )


def extract_hypothesis_and_exp_from_reports(report_file_path: str) -> QlibFactorExperiment | None:
    """
    Extract hypothesis and experiment details from report files.

    Args:
        report_file_path (str): Path to the report file.

    Returns:
        QlibFactorExperiment: An instance of QlibFactorExperiment containing the extracted details.
        None: If no valid experiment is found in the report.
    """
    exp = FactorExperimentLoaderFromPDFfiles().load(report_file_path)
    if exp is None or exp.sub_tasks == []:
        return None

    pdf_screenshot = extract_first_page_screenshot_from_pdf(report_file_path)
    logger.log_object(pdf_screenshot, tag="load_pdf_screenshot")

    docs_dict = load_and_process_pdfs_by_langchain(report_file_path)

    factor_result = {
        task.factor_name: {
            "description": task.factor_description,
            "formulation": task.factor_formulation,
            "variables": task.variables,
            "resources": task.factor_resources,
        }
        for task in exp.sub_tasks
    }

    report_content = "\n".join(docs_dict.values())
    hypothesis = generate_hypothesis(factor_result, report_content)
    exp.hypothesis = hypothesis
    exp.source_report_path = str(Path(report_file_path).resolve())
    exp.source_report_title = Path(report_file_path).stem
    return exp


class FactorReportLoop(FactorRDLoop, metaclass=LoopMeta):
    def __init__(self, report_folder: str = None):
        super().__init__(PROP_SETTING=FACTOR_FROM_REPORT_PROP_SETTING)
        if report_folder is None:
            self.judge_pdf_data_items = json.load(
                open(FACTOR_FROM_REPORT_PROP_SETTING.report_result_json_file_path, "r")
            )
        else:
            self.judge_pdf_data_items = [i for i in Path(report_folder).rglob("*.pdf")]

        self.loop_n = min(len(self.judge_pdf_data_items), FACTOR_FROM_REPORT_PROP_SETTING.report_limit)
        self.shift_report = (
            0  # some reports does not contain viable factor, so we ship some of them to avoid infinite loop
        )

    async def direct_exp_gen(self, prev_out: dict[str, Any]):
        while True:
            if self.get_unfinished_loop_cnt(self.loop_idx) < RD_AGENT_SETTINGS.get_max_parallel():
                report_file_path = self.judge_pdf_data_items[self.loop_idx + self.shift_report]
                logger.info(f"Processing number {self.loop_idx} report: {report_file_path}")
                exp = extract_hypothesis_and_exp_from_reports(str(report_file_path))
                if exp is None:
                    self.shift_report += 1
                    self.loop_n -= 1
                    if self.loop_n < 0:  # NOTE: on every step, we self.loop_n -= 1 at first.
                        raise self.LoopTerminationError("Reach stop criterion and stop loop")
                    continue
                exp.based_experiments = [QlibFactorExperiment(sub_tasks=[], hypothesis=exp.hypothesis)] + [
                    t[0] for t in self.trace.hist if t[1]
                ]
                exp.sub_workspace_list = exp.sub_workspace_list[: FACTOR_FROM_REPORT_PROP_SETTING.max_factors_per_exp]
                exp.sub_tasks = exp.sub_tasks[: FACTOR_FROM_REPORT_PROP_SETTING.max_factors_per_exp]
                exp.base_features = self.plan["features"]
                if exp.based_experiments:
                    exp.based_experiments[-1].base_features = self.plan["features"]
                logger.log_object(exp.hypothesis, tag="hypothesis generation")
                logger.log_object(exp.sub_tasks, tag="experiment generation")
                return exp
            await asyncio.sleep(1)

    def coding(self, prev_out: dict[str, Any]):
        exp = self.coder.develop(prev_out["direct_exp_gen"])
        logger.log_object(exp.sub_workspace_list, tag="coder result")
        return exp

    def record(self, prev_out: dict[str, Any]):
        feedback = prev_out["feedback"]
        exp = prev_out.get("running") or prev_out.get("coding") or prev_out.get("direct_exp_gen")
        exported_count = 0
        if exp is not None and getattr(exp, "prop_dev_feedback", None) is not None:
            source_report_path = getattr(exp, "source_report_path", None)
            source_report_title = getattr(exp, "source_report_title", None)
            for task, workspace, task_feedback in zip(exp.sub_tasks, exp.sub_workspace_list, exp.prop_dev_feedback):
                if workspace is None or task_feedback is None or not bool(task_feedback.final_decision):
                    continue
                try:
                    _, df = workspace.execute("All")
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        f"Failed to reload report-derived factor dataframe for reviewed export: "
                        f"task={task.factor_name}; {exc}"
                    )
                    continue
                if df is None or df.empty:
                    continue
                ic_feedback, full_sample_ic = evaluate_factor_ic_from_workspace(workspace, data_type="All")
                if full_sample_ic is None or abs(full_sample_ic) < FACTOR_COSTEER_SETTINGS.min_abs_ic:
                    logger.info(
                        f"Skip reviewed export for report-derived factor {task.factor_name} because full-sample IC "
                        f"did not pass. {ic_feedback}"
                    )
                    continue
                logic_summary = task.factor_description
                review_notes = "\n".join(
                    part
                    for part in [task_feedback.execution, task_feedback.return_checking, task_feedback.code, ic_feedback]
                    if part
                )
                tags = _infer_report_factor_registry_tags(task, task_feedback)
                workspace.export_reviewed_factor(
                    df,
                    accepted=True,
                    logic_summary=logic_summary,
                    tags=tags,
                    review_notes=review_notes,
                    ic_score=full_sample_ic,
                    source_report_path=source_report_path,
                    source_report_title=source_report_title,
                )
                exported_count += 1

        self.trace.sync_dag_parent_and_hist((exp, feedback), prev_out[self.LOOP_IDX_KEY])
        logger.info(
            f"Factor report loop recorded. Accepted reviewed factor exports: {exported_count}. "
            f"Source report: {getattr(exp, 'source_report_title', 'unknown') if exp is not None else 'unknown'}."
        )


def _infer_report_factor_registry_tags(task, feedback) -> list[str]:
    content = " ".join(
        [
            getattr(task, "factor_name", "") or "",
            getattr(task, "factor_description", "") or "",
            getattr(task, "factor_formulation", "") or "",
            str(getattr(task, "variables", {}) or {}),
            getattr(feedback, "code", "") or "",
        ]
    ).lower()
    tags: set[str] = {"literature_factor", "report_extracted"}
    keyword_to_tag = {
        "momentum": "momentum",
        "reversal": "reversal",
        "volatility": "volatility",
        "volume": "volume",
        "turnover": "turnover",
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
    tags.add("ic_passed")
    return sorted(tags)


def main(report_folder=None, path=None, all_duration=None, checkout=True):
    """
    Auto R&D Evolving loop for fintech factors (the factors are extracted from finance reports).

    Args:
        report_folder (str, optional): The folder contains the report PDF files. Reports will be loaded from this folder.
        path (str, optional): The path for loading a session. If provided, the session will be loaded.
        step_n (int, optional): Step number to continue running a session.
    """
    if path is None and report_folder is None:
        model_loop = FactorReportLoop()
    elif path is not None:
        model_loop = FactorReportLoop.load(path, checkout=checkout)
    else:
        model_loop = FactorReportLoop(report_folder=report_folder)

    asyncio.run(model_loop.run(all_duration=all_duration))


if __name__ == "__main__":
    fire.Fire(main)
