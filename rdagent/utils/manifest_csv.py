"""Helpers for factor manifest CSV (avoid pandas FutureWarning on concat dtype rules)."""

from __future__ import annotations

import pandas as pd


def manifest_append_row(manifest: pd.DataFrame, row: pd.DataFrame) -> pd.DataFrame:
    """Append a one-row manifest DataFrame, aligning columns without all-NA concat quirks."""
    if row.empty:
        return manifest
    cols_order = list(manifest.columns)
    for c in row.columns:
        if c not in cols_order:
            cols_order.append(c)
    left = manifest.reindex(columns=cols_order)
    right = row.reindex(columns=cols_order)
    # Single-row frames often carry all-NA columns for unused manifest fields; pandas warns when
    # concat uses those columns for dtype inference (pandas>=2.1). Drop all-NA columns on the new row.
    right = right.dropna(axis=1, how="all")
    if right.empty:
        return manifest
    # If the manifest is empty, avoid concat with mismatched all-NA columns on the left.
    if left.empty and len(left.columns) == 0:
        return right.copy()
    return pd.concat([left, right], ignore_index=True, copy=False)
