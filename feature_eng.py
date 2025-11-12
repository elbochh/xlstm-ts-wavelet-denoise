# make_features_lgbm_ohlcv.py (with robust yfinance fetch + daily macro merge + MARKET CAP FEATURES + EVENT FLAGS)
# Build LightGBM/xLSTM-ready, leak-free features using OHLCV (+ optional adj_close) + DAILY macro features + DAILY market-cap features + event regime flags.
# Input  : cleaned_ohlcv_30y_all.parquet  +  marketcap_30y.(parquet|csv)
# Output : features_lgbm_ohlcv.parquet

from pathlib import Path
import re
import numpy as np
import pandas as pd
import yfinance as yf

IN_PATH        = Path("cleaned_ohlcv_30y_all.parquet")
OUT_PATH       = Path("features_lgbm_ohlcv.parquet")
# --------------------------- NEW: Market cap input ---------------------------
MARKETCAP_PATH = Path("marketcap_30y_all.parquet")  # will auto-fallback to CSV if parquet not found

# --------------------------- utilities ---------------------------

def to_snake(name: str) -> str:
    name = str(name)
    name = name.replace("^", "")
    name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
    name = re.sub("[^0-9a-zA-Z]+", "_", name)
    name = re.sub("_+", "_", name).strip("_")
    return name.lower()

def snake_case_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols, seen = {}, {}
    for c in df.columns:
        sc = to_snake(c)
        if sc in seen:
            seen[sc] += 1
            sc = f"{sc}_dup{seen[sc]}"
        else:
            seen[sc] = 0
        new_cols[c] = sc
    return df.rename(columns=new_cols)

def safe_div(a, b):
    return a / b.replace({0: np.nan})

def ema(s, span):
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi(price, window=14):
    d = price.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    ag = up.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    al = dn.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100/(1+rs))

def macd(price, fast=12, slow=26, signal=9):
    e_fast  = ema(price, fast)
    e_slow  = ema(price, slow)
    line    = e_fast - e_slow
    signal_ = ema(line, signal)
    hist    = line - signal_
    return line, signal_, hist

def dmi_adx(high, low, close, n=14):
    pc = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-pc).abs(), (low-pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    up_move, down_move = high.diff(), -low.diff()
    pos_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move
    neg_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move
    pos_dm_ema = pos_dm.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    neg_dm_ema = neg_dm.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    pos_di = 100 * safe_div(pos_dm_ema, atr)
    neg_di = 100 * safe_div(neg_dm_ema, atr)
    dx = 100 * safe_div((pos_di - neg_di).abs(), (pos_di + neg_di))
    adx = dx.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    return pos_di, neg_di, adx

def roc(price, n):
    return safe_div(price, price.shift(n)) - 1.0

def kst(price):
    r1 = roc(price, 10).rolling(10, min_periods=10).mean()
    r2 = roc(price, 15).rolling(10, min_periods=10).mean()
    r3 = roc(price, 20).rolling(10, min_periods=10).mean()
    r4 = roc(price, 30).rolling(15, min_periods=15).mean()
    kst_line = 1*r1 + 2*r2 + 3*r3 + 4*r4
    kst_sig  = kst_line.rolling(9, min_periods=9).mean()
    return kst_line, kst_sig

def to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")

def roll_zscore(s, win=21):
    m  = s.rolling(win, min_periods=win).mean()
    sd = s.rolling(win, min_periods=win).std()
    return safe_div(s - m, sd)

# ----------------------- per-ticker OHLCV feature builder -----------------------

ROLL_WINS = [5, 10, 21, 63, 126, 252]

def build_features_per_ticker(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["date"] = pd.to_datetime(g["date"], errors="coerce")
    g = g.dropna(subset=["date"])
    g = g.sort_values("date")

    # Ensure numeric dtypes (accepts adjClose/unadjustedVolume/vwap after snake_case)
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in g.columns:
            raise ValueError(f"Missing required column: {col}")
        g[col] = to_float(g[col])
    if "adj_close" in g.columns:
        g["adj_close"] = to_float(g["adj_close"])

    price = g["adj_close"] if "adj_close" in g.columns else g["close"]
    c, h, l, o = g["close"], g["high"], g["low"], g["open"]
    v = g["volume"]

    # Returns
    g["ret_1d"]     = price.pct_change(1)
    g["ret_log_1d"] = np.log(price).diff()

    # Momentum & volatility windows
    for w in ROLL_WINS:
        g[f"mom_{w}"] = roc(price, w)
    for w in [21, 63, 126]:
        g[f"vol_{w}"] = g["ret_1d"].rolling(w, min_periods=w).std()
    g["vol_21_over_63"]  = safe_div(g.get("vol_21"), g.get("vol_63"))
    g["vol_21_over_126"] = safe_div(g.get("vol_21"), g.get("vol_126"))

    # SMA / EMA
    for w in [10, 20, 50, 100, 200]:
        g[f"sma_{w}"] = price.rolling(w, min_periods=w).mean()
        g[f"ema_{w}"] = ema(price, w)
    g["sma20_over_sma50"]  = safe_div(g["sma_20"], g["sma_50"])
    g["sma50_over_sma200"] = safe_div(g["sma_50"], g["sma_200"])
    g["price_over_sma20"]  = safe_div(price, g["sma_20"])
    g["price_over_sma50"]  = safe_div(price, g["sma_50"])
    g["price_over_sma200"] = safe_div(price, g["sma_200"])

    # EMA slopes / diffs
    g["ema20_diff_1"]          = g["ema_20"] - g["ema_20"].shift(1)
    g["ema50_diff_1"]          = g["ema_50"] - g["ema_50"].shift(1)
    g["ema20_slope_5"]         = (g["ema_20"] - g["ema_20"].shift(5)) / 5.0
    g["ema50_slope_5"]         = (g["ema_50"] - g["ema_50"].shift(5)) / 5.0
    g["ema20_diff_over_price"] = safe_div(g["ema20_diff_1"], price)
    g["ema50_diff_over_price"] = safe_div(g["ema50_diff_1"], price)

    # RSI / Bollinger
    g["rsi_14"] = rsi(price, 14)
    ma20 = price.rolling(20, min_periods=20).mean()
    sd20 = price.rolling(20, min_periods=20).std()
    up20 = ma20 + 2 * sd20
    lo20 = ma20 - 2 * sd20
    g["bb_ma20"]  = ma20
    g["bb_up20"]  = up20
    g["bb_lo20"]  = lo20
    g["bb_pct20"] = (price - lo20) / (up20 - lo20)
    g["bb_bw20"]  = (up20 - lo20) / ma20

    # ATR & ranges
    pc = c.shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    g["atr_14"]     = tr.rolling(14, min_periods=14).mean()
    g["hl_range"]   = safe_div(h - l, l)
    g["co_return"]  = safe_div(c - o, o)
    g["gap_return"] = safe_div(o - pc, pc)

    # Z-scored log return (21d)
    g["ret_log_1d_z21"] = roll_zscore(g["ret_log_1d"], win=21)

    # Stochastic
    ll = l.rolling(14, min_periods=14).min()
    hh = h.rolling(14, min_periods=14).max()
    k  = 100 * (c - ll) / (hh - ll)
    g["stoch_k14"] = k
    g["stoch_d3"]  = k.rolling(3, min_periods=3).mean()

    # MACD / DMI-ADX / KST
    macd_line, macd_sig, macd_hist = macd(price, 12, 26, 9)
    g["macd_line"]   = macd_line
    g["macd_signal"] = macd_sig
    g["macd_hist"]   = macd_hist
    pos_di, neg_di, adx = dmi_adx(h, l, c, n=14)
    g["dmi_pos14"] = pos_di
    g["dmi_neg14"] = neg_di
    g["adx_14"]    = adx
    kst_line, kst_sig = kst(price)
    g["kst_line"]   = kst_line
    g["kst_signal"] = kst_sig

    # Volume features
    g["log_volume"] = np.log1p(v)
    for w in [20, 60, 120]:
        g[f"vol_sma_{w}"] = v.rolling(w, min_periods=w).mean()
        g[f"vol_std_{w}"] = v.rolling(w, min_periods=w).std()
        g[f"vol_z_{w}"]   = safe_div(v - g[f"vol_sma_{w}"], g[f"vol_std_{w}"])
    g["vol_ratio_20"]  = safe_div(v, g["vol_sma_20"])
    g["dollar_volume"] = price * v

    # Cumulative returns
    for w in [21, 63, 126, 252]:
        g[f"cumret_{w}"] = roc(price, w)

    # Calendar (weekday)
    dow = g["date"].dt.weekday
    g["dow_sin"] = np.sin(2*np.pi*dow/7.0)
    g["dow_cos"] = np.cos(2*np.pi+dow*0)/1  # harmless op
    g["dow_cos"] = np.cos(2*np.pi*dow/7.0)  # correct cosine

    return g

# ----------------------- macro (daily) via yfinance -----------------------

MACRO_TICKERS = {
    "^GSPC":     "spx",
    "^VIX":      "vix",
    "CL=F":      "wti",
    "GC=F":      "gold",
    "DX-Y.NYB":  "dxy",
    "EURUSD=X":  "eurusd",
}

def _ensure_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            return x.iloc[:, 0]
        sx = x.squeeze()
        if isinstance(sx, pd.Series):
            return sx
        raise ValueError("Expected 1-D series, got DataFrame with multiple columns.")
    return pd.Series(x)

def fetch_yf_series(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False, group_by="column")
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["date", "px"])
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    if "Date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"Date": "date"})
    if "date" not in df.columns:
        df.rename(columns={df.columns[0]: "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    col_candidates = ["Adj Close", "AdjClose", "Close", "close", "adj_close", "adjclose"]
    px = None
    for col in col_candidates:
        if col in df.columns:
            px = _ensure_series(df[col]); break
    if px is None and isinstance(df.columns, pd.MultiIndex):
        for level_name in ["Adj Close", "Close"]:
            for col in df.columns:
                if isinstance(col, tuple) and col[0] == level_name:
                    px = _ensure_series(df[col]); break
            if px is not None: break
    if px is None:
        numeric_cols = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            px = _ensure_series(df[numeric_cols[-1]])
        else:
            return pd.DataFrame(columns=["date", "px"])
    return pd.DataFrame({"date": df["date"], "px": pd.to_numeric(px, errors="coerce")}).dropna()

def build_macro_panel(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    cal = pd.DataFrame({"date": pd.bdate_range(start=start_date, end=end_date, inclusive="both")})
    macro = cal.copy()
    for tkr, pref in MACRO_TICKERS.items():
        s = fetch_yf_series(tkr, start_date.strftime("%Y-%m-%d"), (end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        if s.empty: continue
        s = s.sort_values("date")
        s[pref + "_close"]     = s["px"]
        s[pref + "_ret_1d"]    = s["px"].pct_change(1)
        s[pref + "_logret_1d"] = np.log(s["px"]).diff()
        s[pref + "_z21"]       = roll_zscore(s["px"], win=21)
        keep = ["date", f"{pref}_close", f"{pref}_ret_1d", f"{pref}_logret_1d", f"{pref}_z21"]
        macro = macro.merge(s[keep], on="date", how="left")
    macro = macro.sort_values("date").ffill()
    for col in [c for c in macro.columns if c != "date"]:
        macro[col + "_lag1"] = macro[col].shift(1)
    return macro
# ---------- Missing-value handling (per-ticker) ----------

def _impute_series_middle(
    s: pd.Series,
    method: str = "nearest",  # 'ffill' (strict past), 'nearest', or 'linear'
    max_gap: int | None = None,  # max consecutive NaNs to fill; None = unlimited
) -> pd.Series:
    """
    Impute only interior NaNs (between first and last valid index). Leave leading/trailing NaNs.
    Assumes s.index is monotonic datetime (per-ticker).
    """
    s = s.copy()
    if s.notna().sum() == 0:
        return s  # nothing to do

    first_idx = s.first_valid_index()
    last_idx  = s.last_valid_index()
    if first_idx is None or last_idx is None or first_idx == last_idx:
        return s  # too few valid points to impute

    # Work on the interior slice only
    interior = s.loc[first_idx:last_idx]

    if method == "ffill":
        filled = interior.ffill()
    elif method == "linear":
        # time-aware linear interpolation
        filled = interior.interpolate(method="time", limit=max_gap, limit_direction="both")
    else:  # 'nearest' (uses both sides)
        # pandas has 'nearest' for index-method interpolation if index is numeric or time
        filled = interior.interpolate(method="nearest", limit=max_gap, limit_direction="both")

    s.loc[first_idx:last_idx] = filled
    return s

def impute_middle_nans_per_ticker(
    df: pd.DataFrame,
    cols_to_impute: list[str],
    date_col: str = "date",
    ticker_col: str = "ticker",
    method: str = "ffill",   # default SAFE (no look-ahead). Use 'nearest' or 'linear' if you prefer.
    max_gap: int | None = 5  # e.g., fill runs up to 5 consecutive NaNs; set None to allow unlimited
        ) -> pd.DataFrame:
    """
    For each ticker, sort by date, impute interior NaNs in `cols_to_impute` using the chosen method.
    Leading/trailing NaNs are left as NaN so they can be dropped later.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.sort_values([ticker_col, date_col])

    def _per_ticker(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy().set_index(date_col)
        for c in cols_to_impute:
            if c in g.columns:
                g[c] = _impute_series_middle(g[c], method=method, max_gap=max_gap)
        return g.reset_index()

    out = (
        out.groupby(ticker_col, group_keys=False, observed=True)
           .apply(_per_ticker)
           .sort_values([ticker_col, date_col])
           .reset_index(drop=True)
    )
    return out

def report_middle_nan_runs(df: pd.DataFrame, cols: list[str], date_col="date", ticker_col="ticker") -> pd.DataFrame:
    """
    Quick diagnostic: counts interior NaNs by ticker & column (ignores leading/trailing).
    """
    rows = []
    for tkr, g in df.sort_values([ticker_col, date_col]).groupby(ticker_col, observed=True):
        g = g.reset_index(drop=True)
        for c in cols:
            s = g[c]
            if s.notna().sum() == 0:
                continue
            first = s.first_valid_index()
            last  = s.last_valid_index()
            if first is None or last is None or first == last:
                continue
            interior = s.iloc[first:last+1]
            rows.append({
                "ticker": tkr,
                "column": c,
                "interior_nan_count": int(interior.isna().sum()),
                "max_consecutive_interior_nans": int(
                    interior.isna().astype(int).groupby((~interior.isna()).cumsum()).sum().max()
                )
            })
    return pd.DataFrame(rows).sort_values(["interior_nan_count","max_consecutive_interior_nans"], ascending=False)


# ----------------------- NEW: event flags (pandemic / war) -----------------------

def build_event_flags(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Returns a bday calendar with regime flags and 1-day-lagged versions.
    Conflicts included (0/1 by trading date):
      - Palestine/Israel:
          * First Intifada:        1987-12-09 .. 1993-09-13
          * Second Intifada:       2000-09-28 .. 2005-02-08
          * Gaza War (2008–09):    2008-12-27 .. 2009-01-18
          * Gaza War (2014):       2014-07-08 .. 2014-08-26
          * Gaza–Israel (2021):    2021-05-10 .. 2021-05-21
          * Israel–Hamas (2023– ): 2023-10-07 .. end_date
      - Region/other with broad market impact:
          * Lebanon War:           2006-07-12 .. 2006-08-14
          * Iraq invasion phase:   2003-03-20 .. 2003-05-01   (initial shock window)
      - Pandemic (unchanged):
          * COVID PHEIC:           2020-01-30 .. 2023-05-05
    """
    # Trading-day calendar
    cal = pd.DataFrame({"date": pd.bdate_range(start=start_date, end=end_date, inclusive="both")})

    # --- Pandemic window (kept as-is) ---
    covid_start = pd.Timestamp("2020-01-30")
    covid_end   = pd.Timestamp("2023-05-05")
    cal["any_pandemic"] = ((cal["date"] >= covid_start) & (cal["date"] <= covid_end)).astype("int8")

    # --- Conflict date ranges ---
    ranges = {
        # Palestine/Israel
        "intifada_first_1987_1993":  (pd.Timestamp("1987-12-09"), pd.Timestamp("1993-09-13")),
        "intifada_second_2000_2005": (pd.Timestamp("2000-09-28"), pd.Timestamp("2005-02-08")),
        "war_gaza_2008":             (pd.Timestamp("2008-12-27"), pd.Timestamp("2009-01-18")),
        "war_gaza_2014":             (pd.Timestamp("2014-07-08"), pd.Timestamp("2014-08-26")),
        "war_gaza_2021":             (pd.Timestamp("2021-05-10"), pd.Timestamp("2021-05-21")),
        "war_israel_hamas_2023":     (pd.Timestamp("2023-10-07"), end_date),

        # Regional / broader
        "war_lebanon_2006":          (pd.Timestamp("2006-07-12"), pd.Timestamp("2006-08-14")),
        "war_iraq_invasion_2003":    (pd.Timestamp("2003-03-20"), pd.Timestamp("2003-05-01")),
    }

    # Create per-conflict flags
    for name, (start, end_) in ranges.items():
        cal[name] = ((cal["date"] >= start) & (cal["date"] <= end_)).astype("int8")

    # Aggregates
    palestine_keys = [
        "intifada_first_1987_1993",
        "intifada_second_2000_2005",
        "war_gaza_2008",
        "war_gaza_2014",
        "war_gaza_2021",
        "war_israel_hamas_2023",
    ]
    all_war_keys = palestine_keys + [
        "war_lebanon_2006",
        "war_iraq_invasion_2003",
    ]

    cal["any_war_palestine"] = (cal[palestine_keys].sum(axis=1) > 0).astype("int8")
    cal["any_war"]           = (cal[all_war_keys].sum(axis=1) > 0).astype("int8")

    # Leak-safe lagged versions (1 trading day)
    for col in ["any_pandemic", "any_war_palestine", "any_war"] + list(ranges.keys()):
        cal[col + "_lag1"] = cal[col].shift(1).fillna(0).astype("int8")

    return cal


# ----------------------- NEW: market-cap feature helpers -----------------------
# Old helpers kept for compatibility (unused in MC block):
def _winsorize_by_date(s: pd.Series, lower=0.01, upper=0.99):
    def _w(x):
        if x.size == 0: return x
        lo, hi = x.quantile([lower, upper])
        return x.clip(lo, hi)
    # NOTE: Series.groupby("date") fails; kept only for backwards compatibility elsewhere
    return s  # no-op fallback

def _zscore_by_date(s: pd.Series):
    return s  # no-op fallback

def _rank_pct_by_date(s: pd.Series):
    return s.rank(pct=True, method="average")  # safe fallback

# --- NEW explicit-key helpers (fix KeyError: 'date'/'ticker') ---
def _winsorize_by_key(s: pd.Series, by, lower=0.01, upper=0.99):
    def _w(x):
        if x.size == 0:
            return x
        lo, hi = x.quantile([lower, upper])
        return x.clip(lo, hi)
    return s.groupby(by, observed=True).transform(_w)

def _zscore_by_key(s: pd.Series, by):
    def _z(x):
        m = x.mean()
        sd = x.std(ddof=0)
        return (x - m) / (sd if sd > 0 else 1.0)
    return s.groupby(by, observed=True).transform(_z)

def _rank_pct_by_key(s: pd.Series, by):
    return s.groupby(by, observed=True).transform(lambda x: x.rank(pct=True, method="average"))

def _group_diff_log(series, lag, by):
    """ ln(x_t) - ln(x_{t-lag}) grouped by `by` (e.g., df['ticker']) """
    s = series.replace([np.inf, -np.inf], np.nan)
    return np.log(s) - np.log(s.groupby(by, observed=True).shift(lag))

def _rolling_sum_by_key(series, by, win):
    return (series.groupby(by, observed=True)
                  .rolling(win, min_periods=win)
                  .sum()
                  .reset_index(level=0, drop=True))

def _rolling_std_by_key(series, by, win):
    return (series.groupby(by, observed=True)
                  .rolling(win, min_periods=win)
                  .std(ddof=0)
                  .reset_index(level=0, drop=True))

def _rolling_mean_by_key(series, by, win):
    return (series.groupby(by, observed=True)
                  .rolling(win, min_periods=win)
                  .mean()
                  .reset_index(level=0, drop=True))

def build_marketcap_features_all(feats: pd.DataFrame) -> pd.DataFrame:
    """
    Expects feats to contain: ['ticker','date','close','volume','market_cap'] + your existing OHLCV features.
    Returns feats with added daily market-cap features (lagged 1 day) and per-date transforms.
    """
    out = feats.copy().sort_values(["ticker","date"])

    # Basic derived
    out["dollar_vol_mc"] = out["close"] * out["volume"]
    out["shares_out"]    = out["market_cap"] / out["close"].replace(0, np.nan)

    # Raw cores (pre-lag)
    out["log_me_raw"] = np.log(out["market_cap"].clip(lower=1.0))
    for k in (21, 63, 126, 252):
        out[f"dlog_me_{k}_raw"] = _group_diff_log(out["market_cap"].clip(lower=1.0), k, by=out["ticker"])

    dlog_me_1 = _group_diff_log(out["market_cap"].clip(lower=1.0), 1, by=out["ticker"])
    out["vol_log_me_63_raw"] = _rolling_std_by_key(dlog_me_1, by=out["ticker"], win=63)

    # Issuance
    out["issue_252_raw"]  = _group_diff_log(out["shares_out"], 252, by=out["ticker"])
    out["issue_1260_raw"] = _group_diff_log(out["shares_out"], 1260, by=out["ticker"])

    # Composite issuance (5y)
    out["logret_mc_1d"]    = np.log(out["close"]).groupby(out["ticker"], observed=True).diff()
    out["sum_logret_1260"] = _rolling_sum_by_key(out["logret_mc_1d"].fillna(0.0), by=out["ticker"], win=1260)
    out["comp_issue_1260_raw"] = _group_diff_log(out["market_cap"].clip(lower=1.0), 1260, by=out["ticker"]) - out["sum_logret_1260"]

    # Liquidity/participation
    out["adv_21_mc"]       = _rolling_mean_by_key(out["dollar_vol_mc"], by=out["ticker"], win=21)
    out["turnover_21_raw"] = _rolling_sum_by_key(out["volume"], by=out["ticker"], win=21) / out["shares_out"]
    out["turnover_63_raw"] = _rolling_sum_by_key(out["volume"], by=out["ticker"], win=63) / out["shares_out"]
    out["mc_over_adv_raw"] = out["market_cap"] / out["adv_21_mc"]

    # --- 1-day lag to avoid look-ahead ---
    raw_cols = [c for c in out.columns if c.endswith("_raw")] + ["adv_21_mc","mc_over_adv_raw"]
    for c in raw_cols:
        base = c.replace("_raw", "")
        out[base] = out.groupby(out["ticker"], observed=True)[c].shift(1)

    # Clean temp columns
    out = out.drop(columns=raw_cols + ["sum_logret_1260"], errors="ignore")

    # --- Per-date cross-sectional transforms (explicit key = date) ---
    feat_cols = [
        "log_me",
        "dlog_me_21","dlog_me_63","dlog_me_126","dlog_me_252",
        "vol_log_me_63",
        "issue_252","issue_1260","comp_issue_1260",
        "turnover_21","turnover_63","mc_over_adv",
    ]
    by_date = out["date"]
    for c in feat_cols:
        out[c + "_w"]   = _winsorize_by_key(out[c], by=by_date)
        out[c + "_z"]   = _zscore_by_key(out[c + "_w"], by=by_date)
        out[c + "_pct"] = _rank_pct_by_key(out[c], by=by_date)

    # Optional dtype compaction
    f64 = out.select_dtypes(include=["float64"]).columns
    out[f64] = out[f64].astype("float32")

    return out

# ------------------------------- NEW: market-cap loader -------------------------------

def load_marketcap_panel() -> pd.DataFrame:
    """
    Loads marketcap_30y as parquet or csv and returns snake_case columns: ['ticker','date','market_cap'].
    """
    if MARKETCAP_PATH.exists():
        mc = pd.read_parquet(MARKETCAP_PATH)
    elif MARKETCAP_PATH.with_suffix(".csv").exists():
        mc = pd.read_csv(MARKETCAP_PATH.with_suffix(".csv"))
    else:
        raise FileNotFoundError("marketcap_30y.(parquet|csv) not found next to script.")
    mc = snake_case_columns(mc)
    # allow common alt names
    rename_map = {}
    if "market_capitalization" in mc.columns and "market_cap" not in mc.columns:
        rename_map["market_capitalization"] = "market_cap"
    if rename_map:
        mc = mc.rename(columns=rename_map)
    req = {"ticker","date","market_cap"}
    missing = req - set(mc.columns)
    if missing:
        raise ValueError(f"Market cap file missing columns: {missing}")
    mc["date"]   = pd.to_datetime(mc["date"], errors="coerce").dt.tz_localize(None)
    mc["ticker"] = mc["ticker"].astype("string")
    mc["market_cap"] = pd.to_numeric(mc["market_cap"], errors="coerce")
    mc = mc.dropna(subset=["date","ticker","market_cap"]).sort_values(["ticker","date"])
    mc = mc.drop_duplicates(["ticker","date"], keep="last")
    return mc


# ------------------------------- main -------------------------------

def main():
    # 1) Load & normalize columns
    df = pd.read_parquet(IN_PATH)
    df = snake_case_columns(df)

    if "label" in df.columns:
        df = df.drop(columns=["label"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad = df["date"].isna().sum()
    if bad:
        print(f"Dropping {bad} row(s) with unparsable date.")
        df = df.dropna(subset=["date"])

    if getattr(df["date"].dt, "tz", None) is not None:
        df["date"] = df["date"].dt.tz_localize(None)

    if "ticker" not in df.columns:
        raise ValueError("Input must include 'ticker' column.")
    df["ticker"] = df["ticker"].astype("string")

    # 2) Merge market cap BEFORE feature building
    try:
        mc = load_marketcap_panel()
        df = df.merge(mc, on=["ticker","date"], how="left")
        print(f"Merged market cap: rows={len(df):,}, missing mc rows={df['market_cap'].isna().sum():,}")
    except Exception as e:
        raise RuntimeError(f"Failed to load/merge market cap file: {e}")

    base_keep = ["ticker", "date", "open", "high", "low", "close", "volume",
                 "adj_close", "unadjusted_volume", "change", "change_percent", "vwap", "change_over_time",
                 "market_cap"]
    keep_cols = [c for c in base_keep if c in df.columns]
    df = df[keep_cols]

    df = df.sort_values(["ticker", "date"]).drop_duplicates(["ticker","date"], keep="last")

    # ---- NEW: impute middle NaNs on raw inputs per ticker (edges stay NaN -> dropped later) ----
    base_impute_cols = [c for c in ["open","high","low","close","volume","adj_close","vwap","market_cap"]
                        if c in df.columns]

    # SAFE (no look-ahead): uses only past values; avoids leaking future into features
    df = impute_middle_nans_per_ticker(
        df,
        cols_to_impute=base_impute_cols,
        method="ffill",     # change to "nearest" or "linear" if you prefer symmetric fill
        max_gap=5           # only bridge short gaps; set None to allow any length
    )

    # Optional: quick diagnostics (comment out in production)
    #diag = report_middle_nan_runs(df, base_impute_cols)
    #print(diag.head(20))


    # 3) Per-ticker OHLCV features (keep ticker column)
    feats = (
        df.groupby("ticker", group_keys=True)
          .apply(build_features_per_ticker, include_groups=False)
          .reset_index(level=0)
          .rename(columns={"level_0": "ticker"})
          .reset_index(drop=True)
    )

    if "market_cap" not in feats.columns:
        feats = feats.merge(df[["ticker","date","market_cap"]], on=["ticker","date"], how="left")

    # 4) Next-day regression target (leak-free)
    feats = feats.sort_values(["ticker","date"])
    feats["y_ret_next"] = np.log(feats.groupby("ticker")["close"].shift(-1)) - np.log(feats["close"])

    # 5) DAILY market-cap features
    req_mc = {"ticker","date","close","volume","market_cap"}
    missing = req_mc - set(feats.columns)
    if missing:
        raise ValueError(f"Missing columns for MC features: {missing}")
    feats["date"] = pd.to_datetime(feats["date"], errors="coerce").dt.tz_localize(None)
    feats = build_marketcap_features_all(feats)

    # 6) Macro merge
    start_date = feats["date"].min()
    end_date   = feats["date"].max()
    print(f"Fetching macro from {start_date.date()} to {end_date.date()} ...")
    macro = build_macro_panel(start_date, end_date)
    feats = feats.merge(macro, on="date", how="left")

    # 7) NEW: Event flags (pandemic/war) + lagged versions
    flags = build_event_flags(start_date, end_date)
    feats = feats.merge(flags, on="date", how="left")

    # 8) Cleanup
    before = len(feats)
    feats = feats.dropna().reset_index(drop=True)
    print(f"Dropped {before - len(feats):,} rows due to NaNs from rolling/macro/event merges.")

    # 9) Downcast
    for col in feats.select_dtypes(include=["float64"]).columns:
        feats[col] = feats[col].astype("float32")
    for col in feats.select_dtypes(include=["int64"]).columns:
        feats[col] = pd.to_numeric(feats[col], downcast="integer")

    # 10) Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(OUT_PATH, engine="pyarrow", compression="snappy", index=False)
    print(f"Saved features (OHLCV + MC + macro + flags): {OUT_PATH} | rows={len(feats):,} cols={feats.shape[1]}")

if __name__ == "__main__":
    main()
