"""
clean_stock_panel.py

Purpose
-------
Clean a multi-ticker OHLCV panel BEFORE modeling to remove odd spikes, illiquid names,
and extreme outliers that can distort training.

What it does (in order, all configurable):
1) Load Parquet (expects at least ['ticker','date', 'adj_close' or 'close']).
2) Compute daily returns per ticker (ret_1d).
3) (Optional) Liquidity filter by rolling dollar volume (drop illiquid rows).
4) (Optional) Clip/winsorize daily returns (global or per-ticker quantiles; plus hard cap).
5) (Optional) Robust z-score filter on returns (drop days > z_max).
6) (Optional) Remove whole tickers with suspicious stats (mean, vol, short history)
   and/or drop top X% most volatile tickers.
7) Save cleaned Parquet + CSV reports of what got removed.

Run
---
$ python clean_stock_panel.py

Outputs
-------
- cleaned_<INPUT>.parquet
- reports/removed_rows.csv         (rows dropped by row-level filters)
- reports/removed_tickers.csv      (tickers removed and why)
- reports/summary.txt              (human-readable summary)
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

# ----------------------------- CONFIG ----------------------------------

# === I/O ===
INPUT_PARQUET  = "ohlcv_30y_all.parquet"     # <- your file
OUTPUT_PARQUET = None                        # if None -> auto: cleaned_<INPUT>.parquet
REPORT_DIR     = "reports"

# === Liquidity filter (row-level) ===
USE_LIQUIDITY_FILTER   = True
DOLLAR_VOL_WINDOW_DAYS = 21
MIN_AVG_DOLLAR_VOL     = 2_000_000.0         # keep rows where rolling avg >= this

# === Return clipping (row-level) ===
USE_RET_CLIP           = True
CLIP_METHOD            = "global"            # "global" or "per_ticker"
LOW_Q, HIGH_Q          = 0.01, 0.99          # quantiles for winsorizing
HARD_ABS_CAP           = 0.20                # also cap to +/-20% after quantiles

# === Robust z-score filter (row-level) ===
USE_Z_FILTER           = True
Z_MAX_ABS              = 5.0                 # drop rows where |z| > 5 based on per-ticker mean/std

# === Ticker-level removal ===
USE_TICKER_STATS_FILTER = True
MIN_OBS_PER_TICKER      = 200                 # require at least this many daily obs
MAX_ABS_MEAN_RET        = 0.05                # drop tickers with |mean daily ret| > 5%
MAX_DAILY_VOL           = 0.15                # drop tickers with daily std > 15%

# Drop the top X% most volatile tickers (by std of ret_1d) across entire panel
DROP_TOP_VOLATILITY_PCT = 0.05                # e.g., 0.05 => drop top 5% most volatile tickers
# Set to 0.0 to disable

# -----------------------------------------------------------------------

@dataclass
class CleanSummary:
    input_rows: int
    input_tickers: int
    after_liquidity_rows: int
    after_clip_rows: int
    after_z_rows: int
    after_ticker_rows: int
    final_rows: int
    final_tickers: int

def _ensure_price_col(df: pd.DataFrame) -> str:
    if "adj_close" in df.columns:
        return "adj_close"
    if "close" in df.columns:
        return "close"
    raise ValueError("Input must have 'adj_close' or 'close'.")

def _ensure_volume_col(df: pd.DataFrame) -> Optional[str]:
    if "volume" in df.columns:
        return "volume"
    if "unadjustedvolume" in df.columns:
        return "unadjustedvolume"
    return None

def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
    return df

def _compute_returns(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    df = df.copy()
    df["ret_1d"] = df.groupby("ticker", sort=False)[price_col].pct_change()
    return df

def _liquidity_filter(df: pd.DataFrame, price_col: str, vol_col: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return filtered df and a DataFrame of removed rows (with reason)."""
    if vol_col is None:
        # No volume available -> skip liquidity filter
        return df.copy(), pd.DataFrame(columns=df.columns.tolist() + ["_remove_reason"])

    work = df.copy()
    work["dollar_vol"] = work[price_col] * work[vol_col]
    work["avg_dollar_vol"] = work.groupby("ticker", sort=False)["dollar_vol"].transform(
        lambda s: s.rolling(DOLLAR_VOL_WINDOW_DAYS, min_periods=DOLLAR_VOL_WINDOW_DAYS).mean()
    )

    mask_keep = work["avg_dollar_vol"] >= float(MIN_AVG_DOLLAR_VOL)
    removed = work.loc[~mask_keep].copy()
    removed["_remove_reason"] = f"liquidity(avg_dollar_vol<{MIN_AVG_DOLLAR_VOL})"
    return work.loc[mask_keep].copy(), removed

def _clip_returns(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Winsorize/clamp ret_1d; return updated df and removed rows (none for clip, but we log pre/post)."""
    work = df.copy()
    if CLIP_METHOD == "global":
        q_low, q_high = work["ret_1d"].quantile([LOW_Q, HIGH_Q])
        work["ret_1d"] = work["ret_1d"].clip(lower=q_low, upper=q_high)
    elif CLIP_METHOD == "per_ticker":
        def _clip(s: pd.Series) -> pd.Series:
            ql, qh = s.quantile([LOW_Q, HIGH_Q])
            return s.clip(lower=ql, upper=qh)
        work["ret_1d"] = work.groupby("ticker", sort=False)["ret_1d"].transform(_clip)
    else:
        raise ValueError("CLIP_METHOD must be 'global' or 'per_ticker'.")

    # Hard absolute cap
    work["ret_1d"] = work["ret_1d"].clip(lower=-HARD_ABS_CAP, upper=HARD_ABS_CAP)

    # We don't "remove" rows here; nothing returned as removed.
    return work, pd.DataFrame(columns=df.columns.tolist() + ["_remove_reason"])

def _z_filter(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Drop rows where per-ticker z-score exceeds Z_MAX_ABS."""
    work = df.copy()

    def _z(s: pd.Series) -> pd.Series:
        mu = s.mean()
        sd = s.std(ddof=0)
        sd = sd if np.isfinite(sd) and sd > 0 else np.nan
        return (s - mu) / (sd + 1e-12)

    work["_z_ret"] = work.groupby("ticker", sort=False)["ret_1d"].transform(_z)
    mask_keep = work["_z_ret"].abs() <= float(Z_MAX_ABS)

    removed = work.loc[~mask_keep].copy()
    removed["_remove_reason"] = f"zscore(|z|>{Z_MAX_ABS})"

    work = work.loc[mask_keep].drop(columns=["_z_ret"])
    return work, removed

def _ticker_stats_filter(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove entire tickers based on stats thresholds and top volatility percentile."""
    work = df.copy()
    stats = work.groupby("ticker")["ret_1d"].agg(mean="mean", std="std", count="count")

    # Threshold flags
    bad = (
        (stats["count"] < MIN_OBS_PER_TICKER) |
        (stats["mean"].abs() > MAX_ABS_MEAN_RET) |
        (stats["std"] > MAX_DAILY_VOL)
    )

    reasons = {}
    if bad.any():
        for tkr, row in stats[bad].iterrows():
            rlist = []
            if row["count"] < MIN_OBS_PER_TICKER:
                rlist.append(f"count<{MIN_OBS_PER_TICKER}")
            if abs(row["mean"]) > MAX_ABS_MEAN_RET:
                rlist.append(f"|mean|>{MAX_ABS_MEAN_RET:.3f}")
            if row["std"] > MAX_DAILY_VOL:
                rlist.append(f"std>{MAX_DAILY_VOL:.3f}")
            reasons[tkr] = ";".join(rlist) if rlist else "stats_threshold"

    # Drop top volatility percentile (optional)
    if DROP_TOP_VOLATILITY_PCT and DROP_TOP_VOLATILITY_PCT > 0:
        cutoff = stats["std"].quantile(1.0 - float(DROP_TOP_VOLATILITY_PCT))
        topvol = stats.index[stats["std"] > cutoff]
        for tkr in topvol:
            reasons.setdefault(tkr, f"top_{int(DROP_TOP_VOLATILITY_PCT*100)}%_vol")
        bad = bad | stats.index.isin(topvol)

    bad_tickers = stats.index[bad]
    removed_rows = work[work["ticker"].isin(bad_tickers)].copy()
    if not removed_rows.empty:
        removed_rows["_remove_reason"] = removed_rows["ticker"].map(reasons).fillna("ticker_removed")

    cleaned = work[~work["ticker"].isin(bad_tickers)].copy()

    # Build ticker-level report
    if len(bad_tickers) > 0:
        tick_report = stats.loc[bad_tickers].copy()
        tick_report["reason"] = tick_report.index.map(reasons)
    else:
        tick_report = pd.DataFrame(columns=["mean", "std", "count", "reason"])

    return cleaned, removed_rows, tick_report

def _write_reports(
    report_dir: Path,
    removed_rows_all: pd.DataFrame,
    removed_tickers_df: pd.DataFrame,
    summary: CleanSummary
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    if not removed_rows_all.empty:
        removed_rows_all.to_csv(report_dir / "removed_rows.csv", index=False)

    if not removed_tickers_df.empty:
        removed_tickers_df.to_csv(report_dir / "removed_tickers.csv", index=True)

    with open(report_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("CLEANING SUMMARY\n")
        for k, v in asdict(summary).items():
            f.write(f"{k}: {v}\n")

def main() -> None:
    in_path = Path(INPUT_PARQUET)
    if not in_path.exists():
        print(f"[ERROR] Input file not found: {in_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(OUTPUT_PARQUET) if OUTPUT_PARQUET else in_path.with_name(f"cleaned_{in_path.name}")
    report_dir = Path(REPORT_DIR)

    print(f"[INFO] Loading: {in_path}")
    df = pd.read_parquet(in_path)
    df = _prep(df)

    price_col = _ensure_price_col(df)
    vol_col   = _ensure_volume_col(df)

    # Initial stats
    input_rows    = len(df)
    input_tickers = df["ticker"].nunique()

    # Compute returns
    df = _compute_returns(df, price_col)
    # Drop first-return NaNs
    df = df.dropna(subset=["ret_1d"]).reset_index(drop=True)

    # Containers for reports
    removed_rows_list = []

    # 1) Liquidity filter
    after_liq_rows = len(df)
    if USE_LIQUIDITY_FILTER:
        df, removed_liq = _liquidity_filter(df, price_col, vol_col)
        removed_rows_list.append(removed_liq)
        after_liq_rows = len(df)
        print(f"[INFO] Liquidity filter kept {after_liq_rows:,} rows.")

    # 2) Clip returns (winsorize + hard cap)
    after_clip_rows = len(df)
    if USE_RET_CLIP:
        df, _ = _clip_returns(df)
        after_clip_rows = len(df)  # (clip doesn't drop rows)
        print(f"[INFO] Clipped returns; rows unchanged at {after_clip_rows:,}.")

    # 3) Robust z-score row removal
    after_z_rows = len(df)
    if USE_Z_FILTER:
        df, removed_z = _z_filter(df)
        removed_rows_list.append(removed_z)
        after_z_rows = len(df)
        print(f"[INFO] Z-filter kept {after_z_rows:,} rows.")

    # 4) Ticker-level removal
    after_ticker_rows = len(df)
    removed_tickers_df = pd.DataFrame()
    if USE_TICKER_STATS_FILTER:
        df, removed_tick_rows, tick_report = _ticker_stats_filter(df)
        removed_rows_list.append(removed_tick_rows)
        removed_tickers_df = tick_report
        after_ticker_rows = len(df)
        print(f"[INFO] Ticker filter kept {after_ticker_rows:,} rows and {df['ticker'].nunique():,} tickers.")

    final_rows    = len(df)
    final_tickers = df["ticker"].nunique()

    # Save cleaned parquet
    df.to_parquet(out_path, index=False)
    print(f"[OK] Saved cleaned data to: {out_path} ({final_rows:,} rows, {final_tickers:,} tickers)")

    # Reports
    removed_rows_all = pd.concat([r for r in removed_rows_list if not r.empty], ignore_index=True) if any(
        (not r.empty) for r in removed_rows_list
    ) else pd.DataFrame(columns=df.columns.tolist() + ["_remove_reason"])

    summary = CleanSummary(
        input_rows=input_rows,
        input_tickers=input_tickers,
        after_liquidity_rows=after_liq_rows,
        after_clip_rows=after_clip_rows,
        after_z_rows=after_z_rows,
        after_ticker_rows=after_ticker_rows,
        final_rows=final_rows,
        final_tickers=final_tickers,
    )
    _write_reports(report_dir, removed_rows_all, removed_tickers_df, summary)

    print("[OK] Reports written to:", report_dir.resolve())
    print("--- SUMMARY ---")
    for k, v in asdict(summary).items():
        print(f"{k:>22}: {v}")

if __name__ == "__main__":
    main()
