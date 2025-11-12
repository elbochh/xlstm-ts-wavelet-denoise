#!/usr/bin/env python3
"""
Leakage sanity for a single rolling window:
- TRAIN = last 6 calendar months (by label date), VAL = next 6 months (most recent)
- Label date (y_date) := next trading date per ticker (aligned with y_ret_next)
- Leakage flags computed ONLY on TRAIN:
    1) near-equality vs target (>= NEAR_EQUAL_FRAC of rows within eps)
    2) high Pearson |corr| >= CORR_FLAG
- Prints flagged features and final feature count, plus split sizes.

Input  : features_lgbm_ohlcv.parquet (your engineered file)
Target : y_ret_next
"""

import numpy as np
import pandas as pd
import pandas.api.types as ptypes

PARQUET_PATH   = "features_lgbm_ohlcv.parquet"
TARGET         = "y_ret_next"

# Leakage thresholds (same spirit as your snippet)
CORR_FLAG        = 0.98      # |corr| above this on TRAIN -> flag
NEAR_EQUAL_FRAC  = 0.90      # >=90% nearly equal to target -> flag
NEAR_EQUAL_EPS   = 1e-12

# IDs / date-like columns to exclude from features
ID_LIKE      = {"id","gvkey","iid","excntry","ticker","permno","permco","cusip","sedol"}
DATE_COLS    = {"date"}      # raw date (we’ll create y_date for splitting)
EXCLUDE_COLS = set()         # add extras here if needed

# ----------------- helpers -----------------

def is_yyyymmdd_like(s: pd.Series, frac_threshold=0.9) -> bool:
    """Detect int YYYYMMDD-like columns to exclude as features (rare for your file)."""
    if not ptypes.is_numeric_dtype(s): return False
    v = pd.to_numeric(s, errors="coerce").dropna()
    if v.empty: return False
    v = v.astype("int64")
    ok = (v >= 19000101) & (v <= 21001231)
    if ok.mean() < frac_threshold: return False
    mm = (v // 100) % 100
    dd = v % 100
    plaus = ok & (mm >= 1) & (mm <= 12) & (dd >= 1) & (dd <= 31)
    return plaus.mean() >= frac_threshold

def build_label_date(df: pd.DataFrame) -> pd.Series:
    """
    y_date = next available 'date' per ticker, aligning with y_ret_next.
    This ensures the split uses the LABEL's timestamp, avoiding leakage.
    """
    d = df[["ticker","date"]].copy()
    d = d.sort_values(["ticker","date"])
    y_date = d.groupby("ticker", sort=False)["date"].shift(-1)
    return y_date

# ----------------- main -----------------

def main():
    df = pd.read_parquet(PARQUET_PATH)
    if TARGET not in df.columns:
        raise RuntimeError(f"Target '{TARGET}' not found. Found columns: {list(df.columns)[:8]}...")

    # Basic types
    if "ticker" not in df.columns or "date" not in df.columns:
        raise RuntimeError("Input must contain 'ticker' and 'date' columns.")
    df["ticker"] = df["ticker"].astype("string")
    df["date"]   = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)

    # Keep only rows with valid target
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET, "date"]).copy()

    # Build label date (y_date) and keep rows with it
    df = df.sort_values(["ticker","date"]).reset_index(drop=True)
    df["y_date"] = build_label_date(df)
    df = df.dropna(subset=["y_date"]).copy()

    # Determine 6M train + 6M val on label date
    max_y   = pd.to_datetime(df["y_date"].max())
    val_start = (max_y - pd.DateOffset(months=6)).replace(day=1)  # start of month for stability
    train_start = (val_start - pd.DateOffset(months=6))
    # Train: [train_start, val_start), Val: [val_start, max_y]
    is_train = (df["y_date"] >= train_start) & (df["y_date"] < val_start)
    is_val   = (df["y_date"] >= val_start) & (df["y_date"] <= max_y)

    df_train = df.loc[is_train].copy()
    df_val   = df.loc[is_val].copy()

    print(f"Date ranges by LABEL date:")
    print(f"  TRAIN: {train_start.date()} .. < {val_start.date()}  rows={len(df_train):,}")
    print(f"  VAL  : {val_start.date()} .. <= {max_y.date()}      rows={len(df_val):,}")

    if len(df_train) < 1000 or len(df_val) < 500:
        print("Warning: very few rows in a split; consider widening the window.")

    # ----------------- Feature candidate set -----------------
    drop_cols = set(EXCLUDE_COLS) | ID_LIKE | DATE_COLS | {"y_date", TARGET}
    # drop non-numeric
    for c in df.columns:
        if c in drop_cols: continue
        if not ptypes.is_numeric_dtype(df[c]):
            drop_cols.add(c)
            continue
        # exclude YYYYMMDD-like ints if any appear
        try:
            if is_yyyymmdd_like(df[c]):
                drop_cols.add(c)
        except Exception:
            pass

    feat_cols = [c for c in df.columns if c not in drop_cols]
    print(f"\nInitial numeric feature count: {len(feat_cols)}")

    # ----------------- Leakage checks on TRAIN ONLY -----------------
    X_tr = df_train[feat_cols].astype("float32")
    y_tr = df_train[TARGET].astype("float32").to_numpy()

    near_equal_drop = []
    corr_drop = []

    # 1) Near-equality
    for c in feat_cols:
        f = X_tr[c].to_numpy()
        m = np.isfinite(f) & np.isfinite(y_tr)
        if m.sum() == 0: 
            continue
        same = np.mean(np.abs(f[m] - y_tr[m]) < NEAR_EQUAL_EPS)
        if same >= NEAR_EQUAL_FRAC:
            near_equal_drop.append(c)

    # 2) High correlation
    for c in feat_cols:
        if c in near_equal_drop:  # already flagged
            continue
        f = X_tr[c].to_numpy()
        m = np.isfinite(f) & np.isfinite(y_tr)
        if m.sum() < 500:  # need enough overlapping points
            continue
        r = np.corrcoef(f[m], y_tr[m])[0, 1]
        if np.isfinite(r) and abs(r) >= CORR_FLAG:
            corr_drop.append(c)

    suspect = sorted(set(near_equal_drop) | set(corr_drop))

    print("\n=== Leakage flags (TRAIN only) ===")
    print(f"Near-equal to target (>= {NEAR_EQUAL_FRAC:.0%} within {NEAR_EQUAL_EPS}): {len(near_equal_drop)}")
    print(f"High correlation (|r| >= {CORR_FLAG}): {len(corr_drop)}")
    if suspect:
        print(f"\nFlagged {len(suspect)} suspicious features (showing up to 40):")
        print(suspect[:40])
    else:
        print("No suspicious features found above thresholds.")

    # Final usable features
    final_feats = [c for c in feat_cols if c not in suspect]
    print(f"\nFinal feature count after leakage filter: {len(final_feats)}")

    # (Optional) Show a few macro vs lag1 examples so you remember to prefer lagged versions
    sample_macro = [c for c in final_feats if c.endswith("_lag1")][:10]
    if sample_macro:
        print("\nSample lagged macro kept:", sample_macro)

    # Save selected features for this window (optional)
    out = pd.DataFrame({"feature": final_feats})
    out.to_csv("selected_features_6m_train.csv", index=False)
    print("\nWrote: selected_features_6m_train.csv")

if __name__ == "__main__":
    main()
