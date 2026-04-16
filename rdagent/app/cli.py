"""
CLI entrance for all rdagent application.

This will
- make rdagent a nice entry and
- autoamtically load dotenv
"""

import os
import sys
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")
# 1) Make sure it is at the beginning of the script so that it will load dotenv before initializing BaseSettings.
# 2) The ".env" argument is necessary to make sure it loads `.env` from the current directory.

import subprocess
from importlib.resources import path as rpath
from typing import Optional

import typer
from typing_extensions import Annotated

from rdagent.scenarios.qlib.knowledge_router import render_route_map

app = typer.Typer()
daily_factor_app = typer.Typer()
minute_factor_app = typer.Typer()
paper_factor_app = typer.Typer()
DEFAULT_PAPER_REPORT_FOLDER = str(Path.cwd() / "papers" / "inbox")
DEFAULT_FACTOR_IMPROVEMENT_FOLDER = str(Path.cwd() / "papers" / "factor_improvement")

CheckoutOption = Annotated[bool, typer.Option("--checkout/--no-checkout", "-c/-C")]
CheckEnvOption = Annotated[bool, typer.Option("--check-env/--no-check-env", "-e/-E")]
CheckDockerOption = Annotated[bool, typer.Option("--check-docker/--no-check-docker", "-d/-D")]
CheckPortsOption = Annotated[bool, typer.Option("--check-ports/--no-check-ports", "-p/-P")]
CheckWorkspaceOption = Annotated[bool, typer.Option("--check-workspace/--no-check-workspace")]


@contextmanager
def _temporary_env(**updates):
    old_values = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _auto_init_workspace() -> None:
    from rdagent.app.utils.init_workspace import init_workspace

    init_workspace(force=False)


def ui(port=19899, log_dir="", debug: bool = False, data_science: bool = False):
    """
    start web app to show the log traces.
    """
    if data_science:
        with rpath("rdagent.log.ui", "dsapp.py") as app_path:
            cmds = ["streamlit", "run", app_path, f"--server.port={port}"]
            subprocess.run(cmds)
        return
    with rpath("rdagent.log.ui", "app.py") as app_path:
        cmds = ["streamlit", "run", app_path, f"--server.port={port}"]
        if log_dir or debug:
            cmds.append("--")
        if log_dir:
            cmds.append(f"--log_dir={log_dir}")
        if debug:
            cmds.append("--debug")
        subprocess.run(cmds)


def server_ui(port=19899):
    """
    start the Flask log server in real time
    """
    from rdagent.log.server.app import main as log_server_main

    log_server_main(port=port)


def ds_user_interact(port=19900):
    """
    start web app to show the log traces in real time
    """
    commands = ["streamlit", "run", "rdagent/log/ui/ds_user_interact.py", f"--server.port={port}"]
    subprocess.run(commands)


@app.command(name="fin_factor")
def fin_factor_cli(
    path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    all_duration: Optional[str] = None,
    checkout: CheckoutOption = True,
):
    from rdagent.app.qlib_rd_loop.factor import main as fin_factor

    fin_factor(path=path, step_n=step_n, loop_n=loop_n, all_duration=all_duration, checkout=checkout)


@app.command(name="fin_model")
def fin_model_cli(
    path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    all_duration: Optional[str] = None,
    checkout: CheckoutOption = True,
):
    from rdagent.app.qlib_rd_loop.model import main as fin_model

    fin_model(path=path, step_n=step_n, loop_n=loop_n, all_duration=all_duration, checkout=checkout)


@app.command(name="fin_quant")
def fin_quant_cli(
    path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    all_duration: Optional[str] = None,
    checkout: CheckoutOption = True,
):
    from rdagent.app.qlib_rd_loop.quant import main as fin_quant

    fin_quant(path=path, step_n=step_n, loop_n=loop_n, all_duration=all_duration, checkout=checkout)


@app.command(name="fin_research")
def fin_research_cli(
    mode: str = "mine_factors",
    path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    all_duration: Optional[str] = None,
    checkout: CheckoutOption = True,
    base_features_path: Optional[str] = None,
    factor_pool_path: Optional[str] = None,
    factor_top_k: Optional[int] = None,
    model_name: Optional[str] = None,
    model_library_path: Optional[str] = None,
    report_file_path: Optional[str] = None,
    auto_rounds: int = 10,
    min_cost_ir: float = 0.0,
    batch_size: int = 10,
):
    from rdagent.app.qlib_rd_loop.research import main as fin_research

    fin_research(
        mode=mode,
        path=path,
        step_n=step_n,
        loop_n=loop_n,
        all_duration=all_duration,
        checkout=checkout,
        base_features_path=base_features_path,
        factor_pool_path=factor_pool_path,
        factor_top_k=factor_top_k,
        model_name=model_name,
        model_library_path=model_library_path,
        report_file_path=report_file_path,
        auto_rounds=auto_rounds,
        min_cost_ir=min_cost_ir,
        batch_size=batch_size,
    )


@app.command(name="fin_mine_factors")
def fin_mine_factors_cli(
    path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    all_duration: Optional[str] = None,
    checkout: CheckoutOption = True,
    base_features_path: Optional[str] = None,
    batch_size: int = 10,
    ):
    from rdagent.app.qlib_rd_loop.research import mine_factors as fin_mine_factors

    _auto_init_workspace()
    fin_mine_factors(
        path=path,
        step_n=step_n,
        loop_n=loop_n,
        all_duration=all_duration,
        checkout=checkout,
        base_features_path=base_features_path,
        batch_size=batch_size,
    )


@app.command(name="daily_factor")
def daily_factor_cli(
    path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    all_duration: Optional[str] = None,
    checkout: CheckoutOption = True,
    base_features_path: Optional[str] = None,
    batch_size: int = 10,
):
    """Mine factors from daily data with the simplest default path."""
    from rdagent.app.qlib_rd_loop.research import mine_factors as fin_mine_factors

    _auto_init_workspace()
    with _temporary_env(RDAGENT_FACTOR_DATA_MODE="daily"):
        fin_mine_factors(
            path=path,
            step_n=step_n,
            loop_n=loop_n,
            all_duration=all_duration,
            checkout=checkout,
            base_features_path=base_features_path,
            batch_size=batch_size,
        )


@app.command(name="minute_factor")
def minute_factor_cli(
    path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    all_duration: Optional[str] = None,
    checkout: CheckoutOption = True,
    base_features_path: Optional[str] = None,
    batch_size: int = 10,
):
    """Mine daily factors from minute and quote sample data."""
    from rdagent.app.qlib_rd_loop.research import mine_factors as fin_mine_factors

    _auto_init_workspace()
    with _temporary_env(RDAGENT_FACTOR_DATA_MODE="minute"):
        fin_mine_factors(
            path=path,
            step_n=step_n,
            loop_n=loop_n,
            all_duration=all_duration,
            checkout=checkout,
            base_features_path=base_features_path,
            batch_size=batch_size,
        )


@daily_factor_app.callback(invoke_without_command=True)
def daily_factor_entry(
    path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    all_duration: Optional[str] = None,
    checkout: CheckoutOption = True,
    base_features_path: Optional[str] = None,
    batch_size: int = 10,
):
    daily_factor_cli(
        path=path,
        step_n=step_n,
        loop_n=loop_n,
        all_duration=all_duration,
        checkout=checkout,
        base_features_path=base_features_path,
        batch_size=batch_size,
    )


@minute_factor_app.callback(invoke_without_command=True)
def minute_factor_entry(
    path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    all_duration: Optional[str] = None,
    checkout: CheckoutOption = True,
    base_features_path: Optional[str] = None,
    batch_size: int = 10,
):
    minute_factor_cli(
        path=path,
        step_n=step_n,
        loop_n=loop_n,
        all_duration=all_duration,
        checkout=checkout,
        base_features_path=base_features_path,
        batch_size=batch_size,
    )


@paper_factor_app.callback(invoke_without_command=True)
def paper_factor_entry(
    report_folder: str = typer.Option(
        DEFAULT_PAPER_REPORT_FOLDER,
        help="Folder containing PDF factor reports",
    ),
    path: Optional[str] = None,
    all_duration: Optional[str] = None,
    checkout: CheckoutOption = True,
):
    paper_factor_cli(
        report_folder=report_folder,
        path=path,
        all_duration=all_duration,
        checkout=checkout,
    )


@app.command(name="fin_model_from_pool")
def fin_model_from_pool_cli(
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    all_duration: Optional[str] = None,
    factor_pool_path: Optional[str] = None,
    factor_top_k: Optional[int] = None,
):
    from rdagent.app.qlib_rd_loop.research import model_from_pool as fin_model_from_pool

    fin_model_from_pool(
        step_n=step_n,
        loop_n=loop_n,
        all_duration=all_duration,
        factor_pool_path=factor_pool_path,
        factor_top_k=factor_top_k,
    )


@app.command(name="fin_lgbm_from_pool")
def fin_lgbm_from_pool_cli(
    factor_pool_path: Optional[str] = None,
    factor_top_k: Optional[int] = None,
):
    from rdagent.app.qlib_rd_loop.research import lgbm_from_pool as fin_lgbm_from_pool

    fin_lgbm_from_pool(
        factor_pool_path=factor_pool_path,
        factor_top_k=factor_top_k,
    )


@app.command(name="fin_backtest_model_library")
def fin_backtest_model_library_cli(
    model_name: Optional[str] = None,
    factor_pool_path: Optional[str] = None,
    factor_top_k: Optional[int] = None,
    model_library_path: Optional[str] = None,
    auto_rounds: int = 10,
    min_cost_ir: float = 0.0,
    llm_decision: bool = True,
):
    from rdagent.app.qlib_rd_loop.research import backtest_model_library as fin_backtest_model_library

    fin_backtest_model_library(
        model_name=model_name,
        factor_pool_path=factor_pool_path,
        factor_top_k=factor_top_k,
        model_library_path=model_library_path,
        auto_rounds=auto_rounds,
        min_cost_ir=min_cost_ir,
        llm_decision=llm_decision,
    )


@app.command(name="fin_import_models_from_report")
def fin_import_models_from_report_cli(
    report_file_path: str,
    model_library_path: Optional[str] = None,
):
    from rdagent.app.qlib_rd_loop.research import main as fin_research

    fin_research(
        mode="paper_to_model_library",
        report_file_path=report_file_path,
        model_library_path=model_library_path,
    )


@app.command(name="fin_list_model_library")
def fin_list_model_library_cli(
    model_library_path: Optional[str] = None,
):
    from rdagent.app.qlib_rd_loop.research import list_model_library as fin_list_model_library

    fin_list_model_library(model_library_path=model_library_path)


@app.command(name="fin_factor_report")
def fin_factor_report_cli(
    report_folder: Optional[str] = None,
    path: Optional[str] = None,
    all_duration: Optional[str] = None,
    checkout: CheckoutOption = True,
):
    from rdagent.app.qlib_rd_loop.factor_from_report import main as fin_factor_report

    _auto_init_workspace()
    fin_factor_report(report_folder=report_folder, path=path, all_duration=all_duration, checkout=checkout)


@app.command(name="paper_factor")
def paper_factor_cli(
    report_folder: str = typer.Option(
        DEFAULT_PAPER_REPORT_FOLDER,
        help="Folder containing PDF factor reports",
    ),
    path: Optional[str] = None,
    all_duration: Optional[str] = None,
    checkout: CheckoutOption = True,
):
    from rdagent.app.qlib_rd_loop.factor_from_report import main as fin_factor_report

    _auto_init_workspace()
    fin_factor_report(report_folder=report_folder, path=path, all_duration=all_duration, checkout=checkout)


@app.command(name="knowledge_map")
def knowledge_map_cli() -> None:
    """Show where each quant-factor workflow step looks for knowledge."""
    typer.echo(render_route_map())


@app.command(name="init")
def init_cli(
    force: bool = typer.Option(False, help="Overwrite existing local workspace files such as .env and factor data."),
) -> None:
    """Create local workspace directories and prepare bundled starter data."""
    from rdagent.app.utils.init_workspace import init_workspace

    summary = init_workspace(force=force)
    typer.echo("RD-Agent workspace initialized.")
    typer.echo(f"Env: {summary['env']}")
    typer.echo("Directories:")
    for path in summary["created_dirs"]:
        typer.echo(f"- {path}")
    typer.echo("Data:")
    for item in summary["data"]:
        typer.echo(f"- {item}")
    typer.echo("Next steps:")
    for item in summary["next_steps"]:
        typer.echo(f"- {item}")


@app.command(name="ingest_factor_papers")
def ingest_factor_papers_cli(
    report_folder: str = typer.Option(
        DEFAULT_FACTOR_IMPROVEMENT_FOLDER,
        help="Folder containing factor-improvement PDF papers",
    ),
) -> None:
    """Ingest factor-improvement papers into the structured paper knowledge base."""
    from rdagent.app.qlib_rd_loop.paper_improvement import main as ingest_factor_improvement_papers

    _auto_init_workspace()
    updated_count = ingest_factor_improvement_papers(report_folder=report_folder)
    typer.echo(f"Ingested {updated_count} paper(s) into the factor-improvement knowledge base.")


@app.command(name="general_model")
def general_model_cli(report_file_path: str):
    from rdagent.app.general_model.general_model import extract_models_and_implement as general_model

    general_model(report_file_path)


@app.command(name="data_science")
def data_science_cli(
    path: Optional[str] = None,
    checkout: CheckoutOption = True,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    timeout: Optional[str] = None,
    competition: Optional[str] = None,
):
    from rdagent.app.data_science.loop import main as data_science

    data_science(
        path=path,
        checkout=checkout,
        step_n=step_n,
        loop_n=loop_n,
        timeout=timeout,
        competition=competition,
    )


@app.command(name="llm_finetune")
def llm_finetune_cli(
    path: Optional[str] = None,
    checkout: CheckoutOption = True,
    benchmark: Optional[str] = None,
    benchmark_description: Optional[str] = None,
    dataset: Optional[str] = None,
    base_model: Optional[str] = None,
    upper_data_size_limit: Optional[int] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    timeout: Optional[str] = None,
):
    from rdagent.app.finetune.llm.loop import main as llm_finetune

    llm_finetune(
        path=path,
        checkout=checkout,
        benchmark=benchmark,
        benchmark_description=benchmark_description,
        dataset=dataset,
        base_model=base_model,
        upper_data_size_limit=upper_data_size_limit,
        step_n=step_n,
        loop_n=loop_n,
        timeout=timeout,
    )


@app.command(name="grade_summary")
def grade_summary_cli(log_folder: str):
    from rdagent.log.mle_summary import grade_summary

    grade_summary(log_folder)


app.command(name="ui")(ui)
app.command(name="server_ui")(server_ui)


@app.command(name="health_check")
def health_check_cli(
    check_env: CheckEnvOption = True,
    check_docker: CheckDockerOption = True,
    check_ports: CheckPortsOption = True,
    check_workspace: CheckWorkspaceOption = True,
):
    from rdagent.app.utils.health_check import health_check

    health_check(
        check_env=check_env,
        check_docker=check_docker,
        check_ports=check_ports,
        check_workspace=check_workspace,
    )


@app.command(name="collect_info")
def collect_info_cli():
    from rdagent.app.utils.info import collect_info

    collect_info()


app.command(name="ds_user_interact")(ds_user_interact)


if __name__ == "__main__":
    app()
