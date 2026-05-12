"""Lightweight checks before paper_factor spends LLM tokens."""

from __future__ import annotations

from pathlib import Path


def verify_paper_factor_data_paths() -> tuple[bool, str]:
    """
    Verify the default local factor data bundle exists and is readable.

    Returns
    -------
    (ok, message)
        ok=False means the run should abort; ok=True with a message starting with WARNING is non-fatal.
    """
    root = Path.cwd() / "git_ignore_folder" / "factor_implementation_source_data"
    daily = root / "daily_pv.h5"
    if not daily.exists():
        return False, f"paper_factor preflight: missing required file `{daily}`. Set cwd to repo root or install data."
    try:
        import pandas as pd

        df = pd.read_hdf(daily, key="data", start=0, stop=5)
    except Exception as exc:  # noqa: BLE001
        return False, f"paper_factor preflight: cannot read `{daily}` (key=data): {exc}"
    if df is None or len(getattr(df, "columns", [])) == 0:
        return False, "paper_factor preflight: daily_pv.h5 has no columns under key=data."

    warnings: list[str] = []
    debug_daily = Path.cwd() / "git_ignore_folder" / "factor_implementation_source_data_debug" / "daily_pv.h5"
    if debug_daily.exists():
        try:
            ddf = pd.read_hdf(debug_daily, key="data", start=0, stop=5)
            full_cols = set(df.columns)
            dbg_cols = set(ddf.columns)
            if full_cols != dbg_cols:
                warnings.append(
                    "paper_factor preflight WARNING: `factor_implementation_source_data` and "
                    "`_debug` daily_pv.h5 column sets differ; Debug vs All evaluations may not be comparable."
                )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"paper_factor preflight WARNING: could not compare debug data columns: {exc}")

    msg = "paper_factor preflight: ok."
    if warnings:
        msg = "\n".join([msg] + warnings)
    return True, msg
