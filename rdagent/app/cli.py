"""
CLI entrance for all rdagent application.

This will
- make rdagent a nice entry and
- autoamtically load dotenv
"""

import os
import shutil
import socket
import sys
import time
import webbrowser
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

app = typer.Typer()
daily_factor_app = typer.Typer()
minute_factor_app = typer.Typer()
paper_factor_app = typer.Typer()
data_app = typer.Typer()
DEFAULT_PAPER_REPORT_FOLDER = str(Path.cwd() / "papers" / "inbox")
DEFAULT_FACTOR_IMPROVEMENT_FOLDER = str(Path.cwd() / "papers" / "factor_improvement")
DEFAULT_FACTOR_PAPER_QUERY = (
    "(cat:q-fin.ST OR cat:q-fin.PM OR cat:q-fin.TR) AND "
    '(all:factor OR all:alpha OR all:predictor OR all:signal OR all:"return prediction" '
    'OR all:"cross-sectional return")'
)

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


def _auto_init_workspace(*, download_missing: bool = True) -> None:
    if download_missing:
        from rdagent.app.utils.init_workspace import init_workspace

        init_workspace(force=False)
    else:
        from rdagent.app.utils.init_workspace import validate_workspace_ready

        validate_workspace_ready(require_factor_data=True)


def _is_local_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _launch_factor_dashboard_server(
    host: str,
    port: int,
    *,
    open_browser_tab: bool,
    report_filter: str | None = None,
) -> str:
    dashboard_url = f"http://{host}:{port}"
    if report_filter:
        from urllib.parse import quote

        dashboard_url = f"{dashboard_url}/?report={quote(report_filter)}&refresh=5"
    else:
        dashboard_url = f"{dashboard_url}/?refresh=5"

    if not _is_local_port_open(host, port):
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "rdagent.app.qlib_rd_loop.factor_dashboard",
                "serve",
                "--host",
                host,
                "--port",
                str(port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        for _ in range(20):
            if _is_local_port_open(host, port):
                break
            time.sleep(0.25)
    if open_browser_tab:
        open_attempts: list[list[str]] = []
        for opener in ("xdg-open", "gio", "gnome-open", "kde-open", "sensible-browser"):
            if shutil.which(opener):
                if opener == "gio":
                    open_attempts.append([opener, "open", dashboard_url])
                else:
                    open_attempts.append([opener, dashboard_url])
        opened = False
        for command in open_attempts:
            try:
                subprocess.Popen(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                opened = True
                break
            except Exception:
                continue
        if not opened:
            try:
                opened = bool(webbrowser.open_new_tab(dashboard_url))
            except Exception:
                opened = False
        if not opened:
            typer.echo(f"Auto-open did not succeed in this environment. Open this URL manually: {dashboard_url}")
    return dashboard_url


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

    _auto_init_workspace(download_missing=False)
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

    _auto_init_workspace(download_missing=False)
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
    """Mine daily factors from minute bar sample data."""
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
    report_file: Optional[str] = typer.Option(
        None,
        help="Specific PDF report to process. If omitted, paper_factor scans the report folder for unprocessed papers.",
    ),
    path: Optional[str] = None,
    all_duration: Optional[str] = None,
    checkout: CheckoutOption = True,
    minimal_mode: bool = typer.Option(
        True,
        "--minimal-mode/--full-mode",
        help="Use the lowest-cost extraction path by skipping report classification and extra hypothesis generation.",
    ),
    auto_fetch: bool = typer.Option(
        False,
        "--auto-fetch/--no-auto-fetch",
        help="Automatically fetch the latest arXiv papers into the report folder before extraction.",
    ),
    fetch_query: str = typer.Option(
        DEFAULT_FACTOR_PAPER_QUERY,
        help="arXiv search query used when --auto-fetch is enabled.",
    ),
    fetch_max_results: int = typer.Option(
        20,
        help="Maximum number of recent arXiv papers to inspect per sync.",
    ),
    fetch_download_limit: Optional[int] = typer.Option(
        None,
        help="Maximum number of new PDFs to download during this sync. Defaults to all new matches.",
    ),
    fetch_days_back: int = typer.Option(
        30,
        help="Only fetch papers submitted within the last N days when --auto-fetch is enabled.",
    ),
    dashboard_host: str = typer.Option("127.0.0.1", help="Host for the factor dashboard server."),
    dashboard_port: int = typer.Option(8765, help="Port for the factor dashboard server."),
    auto_open_dashboard: bool = typer.Option(
        False,
        "--auto-open-dashboard/--no-auto-open-dashboard",
        help="Start the factor dashboard server and open the browser automatically.",
    ),
    llm_max_retry: int = typer.Option(
        1,
        "--llm-max-retry",
        min=1,
        help="Maximum LLM retry count for this paper_factor run. Default 1 to avoid repeating the same LiteLLM error many times.",
    ),
    max_factors_per_paper: int = typer.Option(
        10,
        "--max-factors-per-paper",
        min=1,
        max=10,
        help="Maximum number of representative factors to implement per paper in this run. Default 10.",
    ),
    extract_only: bool = typer.Option(
        False,
        "--extract-only/--run-full-pipeline",
        help="Only read report and extract factor info without coding/evaluation/export.",
    ),
):
    paper_factor_cli(
        report_folder=report_folder,
        report_file=report_file,
        path=path,
        all_duration=all_duration,
        checkout=checkout,
        minimal_mode=minimal_mode,
        auto_fetch=auto_fetch,
        fetch_query=fetch_query,
        fetch_max_results=fetch_max_results,
        fetch_download_limit=fetch_download_limit,
        fetch_days_back=fetch_days_back,
        dashboard_host=dashboard_host,
        dashboard_port=dashboard_port,
        auto_open_dashboard=auto_open_dashboard,
        llm_max_retry=llm_max_retry,
        max_factors_per_paper=max_factors_per_paper,
        extract_only=extract_only,
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
    with _temporary_env(LOG_LLM_CHAT_CONTENT="False"):
        fin_factor_report(report_folder=report_folder, path=path, all_duration=all_duration, checkout=checkout)


@app.command(name="paper_factor")
def paper_factor_cli(
    report_folder: str = typer.Option(
        DEFAULT_PAPER_REPORT_FOLDER,
        help="Folder containing PDF factor reports",
    ),
    report_file: Optional[str] = typer.Option(
        None,
        help="Specific PDF report to process. If omitted, paper_factor scans the report folder for unprocessed papers.",
    ),
    path: Optional[str] = None,
    all_duration: Optional[str] = None,
    checkout: CheckoutOption = True,
    minimal_mode: bool = typer.Option(
        True,
        "--minimal-mode/--full-mode",
        help="Use the lowest-cost extraction path by skipping report classification and extra hypothesis generation.",
    ),
    auto_fetch: bool = typer.Option(
        False,
        "--auto-fetch/--no-auto-fetch",
        help="Automatically fetch the latest arXiv papers into the report folder before extraction.",
    ),
    fetch_query: str = typer.Option(
        DEFAULT_FACTOR_PAPER_QUERY,
        help="arXiv search query used when --auto-fetch is enabled.",
    ),
    fetch_max_results: int = typer.Option(
        20,
        help="Maximum number of recent arXiv papers to inspect per sync.",
    ),
    fetch_download_limit: Optional[int] = typer.Option(
        None,
        help="Maximum number of new PDFs to download during this sync. Defaults to all new matches.",
    ),
    fetch_days_back: int = typer.Option(
        30,
        help="Only fetch papers submitted within the last N days when --auto-fetch is enabled.",
    ),
    dashboard_host: str = typer.Option("127.0.0.1", help="Host for the factor dashboard server."),
    dashboard_port: int = typer.Option(8765, help="Port for the factor dashboard server."),
    auto_open_dashboard: bool = typer.Option(
        False,
        "--auto-open-dashboard/--no-auto-open-dashboard",
        help="Start the factor dashboard server and open the browser automatically.",
    ),
    llm_max_retry: int = typer.Option(
        1,
        "--llm-max-retry",
        min=1,
        help="Maximum LLM retry count for this paper_factor run. Default 1 to avoid repeating the same LiteLLM error many times.",
    ),
    max_factors_per_paper: int = typer.Option(
        10,
        "--max-factors-per-paper",
        min=1,
        max=10,
        help="Maximum number of representative factors to implement per paper in this run. Default 10.",
    ),
    extract_only: bool = typer.Option(
        False,
        "--extract-only/--run-full-pipeline",
        help="Only read report and extract factor info without coding/evaluation/export.",
    ),
):
    _auto_init_workspace(download_missing=False)
    normalized_report_file = str(Path(report_file).resolve()) if report_file else None

    with _temporary_env(
        MAX_RETRY=str(llm_max_retry),
        LOG_LLM_CHAT_CONTENT="False",
        QLIB_FACTOR_MAX_FACTORS_PER_EXP=str(max_factors_per_paper),
        RDAGENT_PAPER_FACTOR_SKIP_LOW_IC_REPAIR="1",
        RDAGENT_PAPER_FACTOR_FAST="1",
    ):
        try:
            from rdagent.app.qlib_rd_loop.factor_from_report import extract_hypothesis_and_exp_from_reports
            from rdagent.app.qlib_rd_loop.factor_from_report import main as fin_factor_report
            from rdagent.app.qlib_rd_loop.factor_from_report import list_unprocessed_report_paths
            from rdagent.app.qlib_rd_loop.paper_fetcher import sync_latest_factor_papers
        except ModuleNotFoundError as exc:
            missing_name = getattr(exc, "name", None) or str(exc)
            typer.echo(
                "paper_factor cannot start because the current Python environment is missing "
                f"the dependency `{missing_name}`.\n"
                "Please activate the project environment or install dependencies first, "
                "then rerun `paper_factor`."
            )
            raise typer.Exit(code=1)

        if auto_open_dashboard:
            dashboard_url = _launch_factor_dashboard_server(
                dashboard_host,
                dashboard_port,
                open_browser_tab=True,
                report_filter=normalized_report_file,
            )
            typer.echo(f"Factor dashboard: {dashboard_url}")

        if path is not None:
            fin_factor_report(
                report_folder=report_folder,
                path=path,
                all_duration=all_duration,
                checkout=checkout,
                minimal_mode=minimal_mode,
            )
            return

        if normalized_report_file:
            report_path = Path(normalized_report_file)
            if not report_path.exists():
                raise typer.BadParameter(f"Report file does not exist: {normalized_report_file}")
            typer.echo(f"Processing demo paper: {normalized_report_file}")
            if extract_only:
                exp = extract_hypothesis_and_exp_from_reports(
                    normalized_report_file,
                    minimal_mode=minimal_mode,
                )
                preview_path = (
                    Path.cwd()
                    / "git_ignore_folder"
                    / "factor_outputs"
                    / "extracted_reports"
                    / f"{report_path.stem}.extracted.json"
                )
                extracted_count = len(exp.sub_tasks) if exp is not None else 0
                typer.echo(f"Extract-only finished. Extracted factors: {extracted_count}")
                typer.echo(f"Preview JSON: {preview_path}")
                if exp is not None and exp.sub_tasks:
                    typer.echo("Extracted factor names:")
                    for task in exp.sub_tasks:
                        typer.echo(f"- {task.factor_name}")
                return
            fin_factor_report(
                report_folder=report_folder,
                all_duration=all_duration,
                checkout=checkout,
                minimal_mode=minimal_mode,
                report_paths=[normalized_report_file],
            )
            typer.echo("paper_factor finished after processing 1 paper.")
            return

        processed_count = 0
        while True:
            local_pending = list_unprocessed_report_paths(report_folder)
            if local_pending:
                next_report = str(local_pending[0])
                typer.echo(f"Processing local pending paper: {next_report}")
                if extract_only:
                    exp = extract_hypothesis_and_exp_from_reports(
                        next_report,
                        minimal_mode=minimal_mode,
                    )
                    preview_path = (
                        Path.cwd()
                        / "git_ignore_folder"
                        / "factor_outputs"
                        / "extracted_reports"
                        / f"{Path(next_report).stem}.extracted.json"
                    )
                    extracted_count = len(exp.sub_tasks) if exp is not None else 0
                    typer.echo(f"Extract-only finished. Extracted factors: {extracted_count}")
                    typer.echo(f"Preview JSON: {preview_path}")
                    processed_count += 1
                    continue
                fin_factor_report(
                    report_folder=report_folder,
                    all_duration=all_duration,
                    checkout=checkout,
                    minimal_mode=minimal_mode,
                    report_paths=[next_report],
                )
                processed_count += 1
                continue

            if not auto_fetch:
                break

            try:
                summary = sync_latest_factor_papers(
                    target_dir=report_folder,
                    query=fetch_query,
                    max_results=fetch_max_results,
                    download_limit=1 if fetch_download_limit is None else min(fetch_download_limit, 1),
                    days_back=fetch_days_back,
                )
            except Exception as exc:  # noqa: BLE001
                typer.echo(f"Auto-fetch failed, stop fetching new papers: {exc}")
                break

            if summary["downloaded_count"] <= 0:
                typer.echo("No new factor papers to fetch.")
                break

            next_report = summary["downloaded_paths"][0]
            typer.echo(f"Fetched and processing paper: {next_report}")
            fin_factor_report(
                report_folder=report_folder,
                all_duration=all_duration,
                checkout=checkout,
                minimal_mode=minimal_mode,
                report_paths=[next_report],
            )
            processed_count += 1

    typer.echo(f"paper_factor finished after processing {processed_count} paper(s).")


@app.command(name="sync_factor_papers")
def sync_factor_papers_cli(
    report_folder: str = typer.Option(
        DEFAULT_PAPER_REPORT_FOLDER,
        help="Folder where downloaded PDF papers will be stored.",
    ),
    fetch_query: str = typer.Option(
        DEFAULT_FACTOR_PAPER_QUERY,
        help="arXiv search query used to retrieve recent papers.",
    ),
    fetch_max_results: int = typer.Option(
        20,
        help="Maximum number of recent arXiv papers to inspect per sync.",
    ),
    fetch_download_limit: Optional[int] = typer.Option(
        None,
        help="Maximum number of new PDFs to download during this sync. Defaults to all new matches.",
    ),
    fetch_days_back: int = typer.Option(
        30,
        help="Only fetch papers submitted within the last N days.",
    ),
) -> None:
    from rdagent.app.qlib_rd_loop.paper_fetcher import sync_latest_factor_papers

    summary = sync_latest_factor_papers(
        target_dir=report_folder,
        query=fetch_query,
        max_results=fetch_max_results,
        download_limit=fetch_download_limit,
        days_back=fetch_days_back,
    )
    typer.echo(f"Downloaded {summary['downloaded_count']} new paper(s) into {summary['target_dir']}")
    typer.echo(f"Failed downloads: {summary.get('failed_count', 0)}")
    typer.echo(f"Manifest: {summary['manifest_path']}")


@app.command(name="knowledge_map")
def knowledge_map_cli() -> None:
    """Show where each quant-factor workflow step looks for knowledge."""
    from rdagent.scenarios.qlib.knowledge_router import render_route_map

    typer.echo(render_route_map())


@app.command(name="init")
def init_cli(
    force: bool = typer.Option(False, help="Overwrite existing local workspace files such as .env and factor data."),
) -> None:
    """Create local workspace directories and prepare bundled starter data."""
    from rdagent.app.utils.init_workspace import init_workspace

    summary = init_workspace(force=force, ingest_factor_improvement=True)
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


@app.command(name="data")
def data_cli(
    force: bool = typer.Option(True, "--force/--if-stale", help="Force a full Tushare refresh instead of only updating stale data."),
) -> None:
    """Refresh daily_pv.h5 and minute_pv.h5 from Tushare."""
    from rdagent.app.utils.tushare_data import auto_update_tushare_data_if_configured

    result = auto_update_tushare_data_if_configured(force=force)
    if result is None:
        typer.echo("TUSHARE_TOKEN is not set. Put it in .env or export it before running this command.")
        raise typer.Exit(code=1)
    typer.echo(result)


@data_app.callback(invoke_without_command=True)
def data_entry(
    force: bool = typer.Option(True, "--force/--if-stale", help="Force a full Tushare refresh instead of only updating stale data."),
) -> None:
    data_cli(force=force)


@app.command(name="update_tushare_data")
def update_tushare_data_cli(
    force: bool = typer.Option(True, "--force/--if-stale", help="Force a full Tushare refresh instead of only updating stale data."),
) -> None:
    """Backward-compatible alias for `rdagent data`."""
    data_cli(force=force)


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
