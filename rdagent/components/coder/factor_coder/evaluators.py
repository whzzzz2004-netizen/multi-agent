import os
import re

from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERMultiFeedback,
    CoSTEERSingleFeedbackDeprecated,
)
from rdagent.components.coder.factor_coder.eva_utils import (
    FactorCodeEvaluator,
    FactorValueEvaluator,
)
from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import Workspace

FactorSingleFeedback = CoSTEERSingleFeedbackDeprecated


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _paper_factor_low_ic_terminal(value_feedback: str | None, code_feedback: str | None) -> bool:
    if not _env_flag("RDAGENT_PAPER_FACTOR_SKIP_LOW_IC_REPAIR"):
        return False
    value_text = (value_feedback or "").lower()
    code_text = (code_feedback or "").lower()
    if "below the minimum absolute ic threshold" not in value_text:
        return False
    structural_failure_markers = [
        "source dataframe is none",
        "more than one column",
        "infinite values. please check",
        "minute-level. this pipeline expects daily",
        "unsupported datetime granularity",
        "failed to evaluate output format",
        "negative shift",
        "negative lookback",
        "negative diff",
        "centered rolling",
        "backward-fills",
    ]
    code_failure_markers = [
        "critic ",
        "critic:",
        "future-information leak",
        "time leakage",
        "traceback",
        "does not align",
        "not aligned",
        "incorrect",
        "wrong",
    ]
    return not any(marker in value_text for marker in structural_failure_markers) and not any(
        marker in code_text for marker in code_failure_markers
    )


class FactorEvaluatorForCoder(CoSTEEREvaluator):
    """This class is the v1 version of evaluator for a single factor implementation.
    It calls several evaluators in share modules to evaluate the factor implementation.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.value_evaluator = FactorValueEvaluator(self.scen)
        self.code_evaluator = FactorCodeEvaluator(self.scen)

    @staticmethod
    def _code_review_passed(code_feedback: str | None) -> bool:
        text = (code_feedback or "").lower()
        if not text.strip():
            return False
        hard_failure_markers = [
            "traceback",
            "execution failed",
            "future-information leak",
            "time leakage",
            "not aligned",
            "does not align",
            "wrong",
            "incorrect",
            "unsupported",
            "unavailable data",
            "missing column",
        ]
        if any(marker in text for marker in hard_failure_markers):
            return False
        return True

    def evaluate(
        self,
        target_task: FactorTask,
        implementation: Workspace,
        gt_implementation: Workspace = None,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> FactorSingleFeedback:
        if implementation is None:
            return None

        target_task_information = target_task.get_task_information()
        if (
            queried_knowledge is not None
            and target_task_information in queried_knowledge.success_task_to_knowledge_dict
        ):
            return queried_knowledge.success_task_to_knowledge_dict[target_task_information].feedback
        elif queried_knowledge is not None and target_task_information in queried_knowledge.failed_task_info_set:
            return FactorSingleFeedback(
                execution_feedback="This task has failed too many times, skip implementation.",
                value_generated_flag=False,
                code_feedback="This task has failed too many times, skip code evaluation.",
                value_feedback="This task has failed too many times, skip value evaluation.",
                final_decision=False,
                final_feedback="This task has failed too many times, skip final decision evaluation.",
                final_decision_based_on_gt=False,
            )
        else:
            factor_feedback = FactorSingleFeedback()
            code_feedback, _ = self.code_evaluator.evaluate(
                target_task=target_task,
                implementation=implementation,
                execution_feedback="No execution was run yet; review the code against the factor definition.",
                value_feedback="No value evaluation has been run yet.",
                gt_implementation=gt_implementation,
            )
            factor_feedback.code_feedback = code_feedback
            if not self._code_review_passed(code_feedback):
                factor_feedback.execution_feedback = "Skipped execution because the code review did not pass."
                factor_feedback.value_feedback = "Skipped value evaluation because the code review did not pass."
                factor_feedback.value_generated_flag = False
                factor_feedback.final_decision_based_on_gt = gt_implementation is not None
                factor_feedback.final_decision = False
                factor_feedback.final_feedback = "Code review failed, rewrite the code."
                return factor_feedback

            # 2. Run the code only after the code review passes.
            (
                execution_feedback,
                gen_df,
            ) = implementation.execute()

            execution_feedback = re.sub(r"(?<=\D)(,\s+-?\d+\.\d+){50,}(?=\D)", ", ", execution_feedback)
            factor_feedback.execution_feedback = "\n".join(
                [line for line in execution_feedback.split("\n") if "warning" not in line.lower()]
            )
            factor_feedback.final_decision_based_on_gt = gt_implementation is not None

            if gen_df is None:
                factor_feedback.value_feedback = "No factor value generated, skip value evaluation."
                factor_feedback.value_generated_flag = False
                factor_feedback.final_decision = False
                factor_feedback.final_feedback = "Execution failed, rewrite the code."
                return factor_feedback

            factor_feedback.value_generated_flag = True
            (
                factor_feedback.value_feedback,
                decision_from_value_check,
            ) = self.value_evaluator.evaluate(
                implementation=implementation, gt_implementation=gt_implementation, version=target_task.version
            )

            if decision_from_value_check is True:
                factor_feedback.final_decision = True
                factor_feedback.final_feedback = "Value evaluation passed, accept the factor."
            else:
                factor_feedback.final_decision = True
                factor_feedback.source_feedback["paper_factor_low_ic_terminal"] = False
                factor_feedback.final_feedback = (
                    "The code review passed and the factor executed successfully, but the IC did not pass; "
                    "recorded as a terminal rejected reproduction."
                )
            return factor_feedback


# TODO:
def shorten_prompt(tpl: str, render_kwargs: dict, shorten_key: str, max_trail: int = 10) -> str:
    """When the prompt is too long. We have to shorten it.
    But we should not truncate the prompt directly, so we should find the key we want to shorten and then shorten it.
    """
    # TODO: this should replace most of code in
    # - FactorFinalDecisionEvaluator.evaluate
    # - FactorCodeEvaluator.evaluate
