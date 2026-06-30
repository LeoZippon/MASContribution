"""Table builders used by scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from mas_contribution_bench.utils.io import ensure_dir


def records_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)


def write_table(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df = records_to_dataframe(records)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_json(path, orient="records", lines=True, force_ascii=False)
