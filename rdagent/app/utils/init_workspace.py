from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from rdagent.log import rdagent_logger as logger


ROOT = Path.cwd()
GIT_IGNORE_DIR = ROOT / "git_ignore_folder"
PAPERS_DIR = ROOT / "papers"
TEMPLATE_DIR = ROOT / "rdagent" / "scenarios" / "qlib" / "experiment" / "factor_data_template"
FACTOR_DATA_DIR = ROOT / "git_ignore_folder" / "factor_implementation_source_data"
FACTOR_DATA_DEBUG_DIR = ROOT / "git_ignore_folder" / "factor_implementation_source_data_debug"
FULL_SAMPLE_DAYS = 252
DEBUG_SAMPLE_DAYS = 60
FULL_SAMPLE_INSTRUMENTS = 80
DEBUG_SAMPLE_INSTRUMENTS = 20


def _copy_if_missing(src: Path, dst: Path, *, force: bool = False) -> bool:
    if not src.exists():
        return False
    if dst.exists() and not force:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    return True


def _is_valid_daily_factor_file(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_hdf(path, key="data")
    except Exception:
        return False
    required_columns = {"$open", "$close", "$high", "$low", "$volume", "$factor"}
    return (
        df is not None
        and not df.empty
        and isinstance(df.index, pd.MultiIndex)
        and set(df.index.names) == {"datetime", "instrument"}
        and required_columns.issubset(df.columns)
    )


def _build_compact_daily_dataset(
    source_path: Path,
    target_path: Path,
    *,
    max_days: int,
    max_instruments: int,
) -> bool:
    if not _is_valid_daily_factor_file(source_path):
        return False
    df = pd.read_hdf(source_path, key="data").sort_index()
    if df.empty:
        return False

    dates = df.index.get_level_values("datetime").unique().sort_values()
    instruments = list(dict.fromkeys(df.index.get_level_values("instrument")))
    chosen_dates = dates[-max_days:]
    chosen_instruments = instruments[:max_instruments]
    compact_df = df.loc[pd.IndexSlice[chosen_dates, chosen_instruments], :].sort_index()
    if compact_df.empty:
        return False

    target_path.parent.mkdir(parents=True, exist_ok=True)
    compact_df.to_hdf(target_path, key="data")
    return True


def _generate_synthetic_daily_factor_file(
    target_path: Path,
    *,
    n_days: int,
    n_instruments: int,
) -> None:
    rng = np.random.default_rng(42)
    end_date = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=end_date, periods=n_days)
    instruments = [f"SH{600000 + i:06d}" for i in range(n_instruments)]
    rows: list[pd.DataFrame] = []

    for idx, instrument in enumerate(instruments):
        base_price = 8.0 + idx * 0.3
        daily_returns = rng.normal(0.0004, 0.018, size=n_days)
        close = base_price * np.cumprod(1 + daily_returns)
        open_ = np.concatenate(([close[0] * (1 - rng.normal(0, 0.004))], close[:-1])) * (
            1 + rng.normal(0, 0.003, size=n_days)
        )
        high = np.maximum(open_, close) * (1 + rng.uniform(0.0005, 0.02, size=n_days))
        low = np.minimum(open_, close) * (1 - rng.uniform(0.0005, 0.02, size=n_days))
        volume = rng.lognormal(mean=14.0, sigma=0.5, size=n_days) * (1 + 0.15 * np.sin(np.linspace(0, 6, n_days)))
        factor = np.clip(1 + np.cumsum(rng.normal(0.0, 0.0008, size=n_days)), 0.7, 1.3)
        instrument_index = pd.MultiIndex.from_product([dates, [instrument]], names=["datetime", "instrument"])
        rows.append(
            pd.DataFrame(
                {
                    "$open": open_.astype(np.float32),
                    "$close": close.astype(np.float32),
                    "$high": high.astype(np.float32),
                    "$low": low.astype(np.float32),
                    "$volume": volume.astype(np.float32),
                    "$factor": factor.astype(np.float32),
                },
                index=instrument_index,
            )
        )

    synthetic_df = pd.concat(rows).sort_index()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_df.to_hdf(target_path, key="data")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _minute_day_count(path: Path) -> int:
    if not path.exists():
        return 0
    df = pd.read_hdf(path, key="data")
    if df.empty or "datetime" not in df.index.names:
        return 0
    dt = pd.to_datetime(df.index.get_level_values("datetime"))
    return int(dt.normalize().nunique())


def _minute_timestamps_for_day(day: pd.Timestamp) -> pd.DatetimeIndex:
    morning = pd.date_range(day.normalize() + pd.Timedelta(hours=9, minutes=30), periods=120, freq="min")
    afternoon = pd.date_range(day.normalize() + pd.Timedelta(hours=13), periods=120, freq="min")
    return morning.append(afternoon)


def _build_intraday_profile(minutes_per_day: int = 240) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-1.0, 1.0, minutes_per_day)
    volume_curve = 0.55 + 0.45 * np.abs(x)
    volume_curve = volume_curve / volume_curve.sum()
    drift_curve = np.linspace(0.0, 1.0, minutes_per_day)
    return volume_curve, drift_curve


def _generate_minute_sample_data(
    daily_path: Path,
    minute_pv_path: Path,
    minute_quote_path: Path,
    max_days: int,
    max_instruments: int,
) -> None:
    daily_df = pd.read_hdf(daily_path, key="data").sort_index()
    instruments = list(dict.fromkeys(daily_df.index.get_level_values("instrument")))
    chosen_instruments = instruments[:max_instruments]
    available_dates = daily_df.index.get_level_values("datetime").unique().sort_values()
    chosen_dates = available_dates[-max_days:]
    sampled_daily = daily_df.loc[pd.IndexSlice[chosen_dates, chosen_instruments], :].copy()

    volume_curve, drift_curve = _build_intraday_profile()
    minute_frames: list[pd.DataFrame] = []
    quote_frames: list[pd.DataFrame] = []

    for (dt, instrument), row in sampled_daily.iterrows():
        base_open = float(row["$open"])
        base_close = float(row["$close"])
        base_high = float(row["$high"])
        base_low = float(row["$low"])
        base_volume = max(float(row["$volume"]), 0.0)
        minute_index = pd.MultiIndex.from_product(
            [_minute_timestamps_for_day(pd.Timestamp(dt)), [instrument]],
            names=["datetime", "instrument"],
        )

        trend = base_open + (base_close - base_open) * drift_curve
        oscillation = 0.12 * (base_high - base_low + 1e-6) * np.sin(np.linspace(0.0, 4.0 * np.pi, len(drift_curve)))
        mid = np.clip(trend + oscillation, base_low, base_high)

        prev_close = np.concatenate(([base_open], mid[:-1]))
        close = mid
        high = np.clip(np.maximum(prev_close, close) + 0.05 * (base_high - base_low + 1e-6), base_low, base_high)
        low = np.clip(np.minimum(prev_close, close) - 0.05 * (base_high - base_low + 1e-6), base_low, base_high)

        minute_volume = np.maximum(np.round(base_volume * volume_curve), 0.0)
        if minute_volume.sum() > 0:
            minute_volume[-1] += base_volume - minute_volume.sum()
        vwap = (prev_close + high + low + close) / 4.0

        spread = np.maximum(np.abs(close) * 0.0005, 1e-4)
        bid1 = close - spread / 2.0
        ask1 = close + spread / 2.0
        sinusoid = np.sin(np.linspace(0, 2 * np.pi, len(close)))
        bid1_size = np.maximum(np.round(minute_volume * (0.45 + 0.1 * sinusoid)), 1.0)
        ask1_size = np.maximum(np.round(minute_volume * (0.55 - 0.1 * sinusoid)), 1.0)

        minute_frames.append(
            pd.DataFrame(
                {
                    "$open": prev_close,
                    "$close": close,
                    "$high": high,
                    "$low": low,
                    "$volume": minute_volume,
                    "$vwap": vwap,
                },
                index=minute_index,
            )
        )
        quote_frames.append(
            pd.DataFrame(
                {
                    "$bid1": bid1,
                    "$ask1": ask1,
                    "$bid1_size": bid1_size,
                    "$ask1_size": ask1_size,
                    "$mid_price": (bid1 + ask1) / 2.0,
                    "$spread_bps": ((ask1 - bid1) / np.maximum((bid1 + ask1) / 2.0, 1e-8)) * 10000.0,
                },
                index=minute_index,
            )
        )

    pd.concat(minute_frames).sort_index().to_hdf(minute_pv_path, key="data")
    pd.concat(quote_frames).sort_index().to_hdf(minute_quote_path, key="data")


def _ensure_minute_data_files() -> None:
    for folder, max_days, max_instruments in [
        (FACTOR_DATA_DIR, 252, 80),
        (FACTOR_DATA_DEBUG_DIR, 60, 20),
    ]:
        daily_path = folder / "daily_pv.h5"
        minute_pv_path = folder / "minute_pv.h5"
        minute_quote_path = folder / "minute_quote.h5"
        if not daily_path.exists():
            continue
        if (
            minute_pv_path.exists()
            and minute_quote_path.exists()
            and _minute_day_count(minute_pv_path) >= max_days
        ):
            continue
        _generate_minute_sample_data(daily_path, minute_pv_path, minute_quote_path, max_days, max_instruments)


def _ensure_workspace_dirs() -> list[Path]:
    dirs = [
        GIT_IGNORE_DIR,
        GIT_IGNORE_DIR / "factor_outputs",
        GIT_IGNORE_DIR / "research_store" / "knowledge",
        GIT_IGNORE_DIR / "research_store" / "knowledge_v2" / "paper_improvement",
        GIT_IGNORE_DIR / "research_store" / "knowledge_v2" / "error_cases",
        GIT_IGNORE_DIR / "factor_implementation_source_data",
        GIT_IGNORE_DIR / "factor_implementation_source_data_debug",
        GIT_IGNORE_DIR / "RD-Agent_workspace",
        GIT_IGNORE_DIR / "traces",
        GIT_IGNORE_DIR / "static",
        PAPERS_DIR,
        PAPERS_DIR / "inbox",
        PAPERS_DIR / "factor_improvement",
    ]
    for path in dirs:
        _ensure_dir(path)
    return dirs


def _ensure_env_file(force: bool = False) -> str:
    env_path = ROOT / ".env"
    env_example_path = ROOT / ".env.example"
    if env_path.exists() and not force:
        return f"Kept existing env file: {env_path}"
    if env_example_path.exists():
        shutil.copy(env_example_path, env_path)
        return f"Created env file from template: {env_path}"
    return "Skipped env file creation because .env.example is missing."


def _ensure_factor_data(force: bool = False) -> list[str]:
    actions: list[str] = []
    full_data_dir = FACTOR_DATA_DIR
    debug_data_dir = FACTOR_DATA_DEBUG_DIR
    _ensure_dir(full_data_dir)
    _ensure_dir(debug_data_dir)
    full_template_path = TEMPLATE_DIR / "daily_pv_all.h5"
    debug_template_path = TEMPLATE_DIR / "daily_pv_debug.h5"
    full_target_path = full_data_dir / "daily_pv.h5"
    debug_target_path = debug_data_dir / "daily_pv.h5"

    full_method = None
    debug_method = None
    if force or not _is_valid_daily_factor_file(full_target_path):
        built_from_template = _build_compact_daily_dataset(
            full_template_path,
            full_target_path,
            max_days=FULL_SAMPLE_DAYS,
            max_instruments=FULL_SAMPLE_INSTRUMENTS,
        )
        if built_from_template:
            full_method = "template_subset"
        else:
            _generate_synthetic_daily_factor_file(
                full_target_path,
                n_days=FULL_SAMPLE_DAYS,
                n_instruments=FULL_SAMPLE_INSTRUMENTS,
            )
            full_method = "synthetic"
    if force or not _is_valid_daily_factor_file(debug_target_path):
        built_from_template = _build_compact_daily_dataset(
            debug_template_path,
            debug_target_path,
            max_days=DEBUG_SAMPLE_DAYS,
            max_instruments=DEBUG_SAMPLE_INSTRUMENTS,
        )
        if built_from_template:
            debug_method = "template_subset"
        else:
            if _is_valid_daily_factor_file(full_target_path):
                _build_compact_daily_dataset(
                    full_target_path,
                    debug_target_path,
                    max_days=DEBUG_SAMPLE_DAYS,
                    max_instruments=DEBUG_SAMPLE_INSTRUMENTS,
                )
                debug_method = "full_subset"
            else:
                _generate_synthetic_daily_factor_file(
                    debug_target_path,
                    n_days=DEBUG_SAMPLE_DAYS,
                    n_instruments=DEBUG_SAMPLE_INSTRUMENTS,
                )
                debug_method = "synthetic"
    copied_full_readme = _copy_if_missing(TEMPLATE_DIR / "README.md", full_data_dir / "README.md", force=force)
    copied_debug_readme = _copy_if_missing(TEMPLATE_DIR / "README.md", debug_data_dir / "README.md", force=force)

    full_ready = _is_valid_daily_factor_file(full_target_path)
    debug_ready = _is_valid_daily_factor_file(debug_target_path)

    if not (full_ready and debug_ready):
        raise RuntimeError(
            "Failed to prepare compact factor data during init. "
            f"Expected valid files at {full_target_path} and {debug_target_path}."
        )

    if full_method is not None:
        actions.append(f"Prepared compact daily factor data ({full_method}): {full_target_path}")
    elif full_ready:
        actions.append(f"Using daily factor data: {full_target_path}")
    if debug_method is not None:
        actions.append(f"Prepared compact debug daily factor data ({debug_method}): {debug_target_path}")
    elif debug_ready:
        actions.append(f"Using debug daily factor data: {debug_target_path}")
    if copied_full_readme or copied_debug_readme:
        actions.append("Prepared factor data README files.")

    _ensure_minute_data_files()
    actions.append(f"Prepared minute factor data: {full_data_dir / 'minute_pv.h5'}")
    actions.append(f"Prepared debug minute factor data: {debug_data_dir / 'minute_pv.h5'}")
    return actions


def init_workspace(force: bool = False) -> dict[str, list[str] | str]:
    created_dirs = [str(path) for path in _ensure_workspace_dirs()]
    env_message = _ensure_env_file(force=force)
    data_actions = _ensure_factor_data(force=force)

    summary = {
        "created_dirs": created_dirs,
        "env": env_message,
        "data": data_actions,
        "next_steps": [
            "Review .env and fill in your API keys if needed.",
            "Run `rdagent health_check --no-check-docker` to verify API and port configuration.",
            "Run `rdagent daily_factor` or `rdagent minute_factor` to start mining factors.",
        ],
    }
    logger.info("Workspace initialization finished.")
    return summary
