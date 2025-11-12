# cluster_stocks_by_vol_mcap.py
# Cluster stocks by recent (trailing) market cap & volatility for per-cluster training

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = Path("features_lgbm_ohlcv.parquet")
OUTPUT_FILE   = "ohlcv_clusters.csv"

# Controls
N_CLUSTERS     = 2
USE_ADJ_CLOSE  = True
VOL_LAMBDA     = 0.94   # EWMA decay for daily returns (RiskMetrics-style)
VOL_LOOKBACK   = 63     # days for EWMA seed (63 or 126 are common)
MCAP_LOOKBACK  = 252    # trailing days for average market cap
ANNUALIZE_VOL  = True
SCALE_FEATURES = True

def ewma_vol(log_ret: pd.Series, lam=VOL_LAMBDA) -> float:
    """EWMA volatility estimator over available series (returns daily sigma)."""
    # drop NaNs
    r = log_ret.dropna().values
    if r.size == 0:
        return 0.0
    # initialize with sample var of first chunk (up to VOL_LOOKBACK)
    init_n = min(VOL_LOOKBACK, r.size)
    v = np.var(r[:init_n], ddof=0) if init_n > 1 else (r[0]**2 if r.size else 0.0)
    for x in r[init_n:]:
        v = lam * v + (1 - lam) * (x * x)
    return float(np.sqrt(v))

def load_panel() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH)
    df.columns = [c.lower() for c in df.columns]
    need = {"ticker", "date", "close", "market_cap"}
    if not need.issubset(df.columns):
        missing = need - set(df.columns)
        raise ValueError(f"{FEATURES_PATH} missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    if USE_ADJ_CLOSE and "adj_close" in df.columns:
        df["price"] = pd.to_numeric(df["adj_close"], errors="coerce")
    else:
        df["price"] = pd.to_numeric(df["close"], errors="coerce")
    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
    df = df.dropna(subset=["ticker", "date", "price", "market_cap"])
    df = df[df["price"] > 0].sort_values(["ticker", "date"])
    # daily log returns per ticker
    df["log_ret"] = np.log(df["price"]) - np.log(df.groupby("ticker", observed=True)["price"].shift(1))
    return df

def build_snapshot(df: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    # limit to trailing windows ending at ref_date (per ticker)
    df = df[df["date"] <= ref_date].copy()

    def per_ticker(g: pd.DataFrame) -> pd.Series:
        # trailing windows
        g = g.sort_values("date")
        g_tail_mcap = g.tail(MCAP_LOOKBACK)
        g_tail_ret  = g.tail(max(MCAP_LOOKBACK, VOL_LOOKBACK))  # enough history for EWMA seed

        # features
        mcap_252_mean = g_tail_mcap["market_cap"].mean()
        vol_ewma = ewma_vol(g_tail_ret["log_ret"], lam=VOL_LAMBDA)

        if ANNUALIZE_VOL:
            vol_ewma *= np.sqrt(252.0)

        return pd.Series({
            "log_mcap_252d_mean": np.log1p(mcap_252_mean) if pd.notna(mcap_252_mean) else 0.0,
            "vol_ewma": 0.0 if pd.isna(vol_ewma) else vol_ewma,
            "n_obs": len(g),
            "last_date": g["date"].max()
        })

    agg = df.groupby("ticker", observed=True).apply(per_ticker).reset_index()
    return agg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_date", type=str, default=None,
                        help="Reference date (YYYY-MM-DD). Defaults to latest date in data.")
    parser.add_argument("--n_clusters", type=int, default=N_CLUSTERS)
    args = parser.parse_args()

    df = load_panel()
    ref_date = pd.to_datetime(args.ref_date).tz_localize(None) if args.ref_date else df["date"].max()
    print(f"Reference date for clustering: {ref_date.date()}")

    snap = build_snapshot(df, ref_date)

    # Filter tickers with too little history (e.g., < 126 days)
    snap = snap[snap["n_obs"] >= max(126, VOL_LOOKBACK)].copy()

    X = snap[["log_mcap_252d_mean", "vol_ewma"]].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if SCALE_FEATURES:
        X = StandardScaler().fit_transform(X)
    else:
        X = X.values

    try:
        km = KMeans(n_clusters=args.n_clusters, random_state=42, n_init="auto")
    except TypeError:
        km = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)

    snap["cluster"] = km.fit_predict(X)
    out = snap[["ticker", "cluster"]].sort_values(["cluster", "ticker"])
    out.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {len(out):,} tickers to {OUTPUT_FILE}")
    med = snap.groupby("cluster")[["log_mcap_252d_mean", "vol_ewma"]].median()
    print("\nCluster medians:\n", med.to_string())

if __name__ == "__main__":
    main()
