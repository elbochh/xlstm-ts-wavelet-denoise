"""
Expanding-window xLSTM regression + Wavelet denoise + GMADL (your version)
- Cluster-aware: one model per cluster (ohlcv_clusters.csv)
- Windows:
    TRAIN  = expanding history (ALL before eval_start), but first window requires >= INITIAL_TRAIN_YEARS
    VALID  = eval_start → val_end  (VAL_MONTHS)
    TEST   = val_end    → test_end (TEST_MONTHS)
- FS on TRAIN ONLY (optionally last FS_LOOKBACK_MONTHS of TRAIN)
- Early stop on VALID RMSE (CSV schema preserved)
- Live epoch logs + per-window feature list
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# --- Feature selection backends ---
try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

# --- Optional denoise ---
try:
    import pywt
    def wavelet_denoise_1d(x, wavelet="db4", level=None):
        x = np.asarray(x, dtype=np.float32)
        finite = np.isfinite(x)
        if finite.sum() < 8:
            return x
        if not finite.all():
            m = np.nanmean(x[finite]) if finite.any() else 0.0
            x = np.where(finite, x, m)
        coeffs = pywt.wavedec(x, wavelet=wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745 + 1e-8
        uth = sigma * np.sqrt(2 * np.log(len(x)))
        den = [coeffs[0]] + [pywt.threshold(c, value=uth, mode="soft") for c in coeffs[1:]]
        return pywt.waverec(den, wavelet=wavelet)[: len(x)]
    HAS_WAVELET = True
except Exception:
    HAS_WAVELET = False

# ---------------- CONFIG (edit here) ----------------

PARQUET_PATH   = "features_lgbm_ohlcv.parquet"
CLUSTERS_CSV   = "ohlcv_clusters.csv"          # columns: ticker,cluster
OUTPUT_PREFIX  = "rolling_results_lookback"    # -> <prefix>_cluster{c}.csv

# Windowing (expanding history after an initial minimum)
INITIAL_TRAIN_YEARS = 10       # require at least this many years before first eval window
VAL_MONTHS          = 6
TEST_MONTHS         = 2
STEP_MONTHS         = TEST_MONTHS  # advance by the test block
USE_EXPANDING       = True         # True => TRAIN uses ALL history before eval_start

# Feature selection
FS_LOOKBACK_MONTHS  = 18           # FS on last N months of TRAIN (None to use all TRAIN)
FS_BACKEND          = "lgbm"       # "lgbm" | "rf" | "variance"
TOP_K               = 31

# Modeling (defaults; can be overridden per-cluster below)
SEQ_LEN      = 64
BATCH_SIZE   = 256
EPOCHS       = 15
PATIENCE     = 3
LR           = 1e-3
WEIGHT_DECAY = 1e-4
D_MODEL      = 20
LAYERS       = 1
DROPOUT      = 0.3
DENOISE      = True

# === Your GMADL parameters (match obj_gmadl / loss_gmadl) ===
ABS_EPS         = 1e-3     # smooth-abs epsilon
HESS_EPS        = 1e-9     # kept for parity; not used directly in torch loss
SIG_K           = 15.0     # logistic slope
GMADL_MARGIN    = 0.0      # margin on y*ŷ
GMADL_LAMBDA    = 0.5      # penalty weight λ

# Per-cluster overrides (optional)
# CLUSTER_CONFIG = { 0: {"TOP_K": 40, "LR": 8e-4}, 1: {"TOP_K": 28, "FS_BACKEND": "rf"} }
CLUSTER_CONFIG: Dict[int, Dict[str, object]] = {}

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else ("mps" if torch.backends.mps.is_available() else "cpu"))
print("Device:", DEVICE)

# ------------------------ Utilities ------------------------

def normalize_columns_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: c.strip().lower().replace(" ", "_") for c in df.columns})

def month_floor(d: pd.Timestamp) -> pd.Timestamp:
    d = pd.to_datetime(d)
    return pd.Timestamp(d.year, d.month, 1)

# ------------------------ Dataset ------------------------

class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ------------------------ Model ------------------------

class EMAPath(nn.Module):
    def __init__(self, d_in: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.gate = nn.Linear(d_in, d_model)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        z = self.proj(x)
        g = self.sigmoid(self.gate(x))
        y = []
        prev = torch.zeros_like(z[:, 0, :])
        for t in range(z.size(1)):
            prev = g[:, t, :] * z[:, t, :] + (1 - g[:, t, :]) * prev
            y.append(prev)
        return torch.stack(y, dim=1)

class XLSTMRegressor(nn.Module):
    def __init__(self, d_in: int, d_model: int = 64, layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=layers, batch_first=True,
                            dropout=(dropout if layers > 1 else 0.0))
        self.ema = EMAPath(d_in, d_model)
        self.norm = nn.LayerNorm(d_model * 2)
        self.head = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d_model, 1))
    def forward(self, x):
        x_proj = self.input_proj(x)
        lstm_out, _ = self.lstm(x_proj)
        ema_out = self.ema(x)
        last = torch.cat([lstm_out[:, -1, :], ema_out[:, -1, :]], dim=-1)
        last = self.norm(last)
        return self.head(last).squeeze(-1)

# ------------------------ Your GMADL (torch) ------------------------
# Mirrors obj_gmadl / loss_gmadl: smooth_abs(|y-ŷ|) + λ·|y|·sigmoid(SIG_K·(M − y·ŷ))

class GMADL_Torch(nn.Module):
    def __init__(self, lam=GMADL_LAMBDA, margin=GMADL_MARGIN, sig_k=SIG_K, abs_eps=ABS_EPS):
        super().__init__()
        self.lam = lam
        self.margin = margin
        self.sig_k = sig_k
        self.abs_eps = abs_eps
    def smooth_abs(self, e: torch.Tensor) -> torch.Tensor:
        # sqrt(e^2 + eps^2)  (gradient: e / sqrt(...))  -> matches g1 in your obj
        return torch.sqrt(e * e + (self.abs_eps ** 2))
    def forward(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = y.view_as(yhat)
        e = yhat - y
        base = self.smooth_abs(e)
        margin_term = self.sig_k * (self.margin - y * yhat)
        pen = self.lam * torch.abs(y) * torch.sigmoid(margin_term)
        return (base + pen).mean()

# ------------------------ Helpers ------------------------

def build_y_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["y_date"] = df.groupby("ticker")["date"].shift(-1)
    return df.dropna(subset=["y_ret_next", "y_date"]).reset_index(drop=True)

def lgbm_feature_select(train_df, feature_cols, target_col, top_k, backend="lgbm"):
    X = train_df[feature_cols].astype(np.float32)
    y = train_df[target_col].astype(np.float32)
    if backend == "lgbm" and HAS_LGBM:
        model = lgb.LGBMRegressor(
            n_estimators=800, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1
        )
        model.fit(X, y)
        imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        return list(imp.index[:top_k])
    elif backend == "rf":
        rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=1)
        rf.fit(X, y)
        imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
        return list(imp.index[:top_k])
    else:
        return list(X.var().sort_values(ascending=False).index[:top_k])

def denoise_split(df, cols):
    if not HAS_WAVELET or not DENOISE:
        return df
    out = []
    for _, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        for c in cols:
            try:
                g[c] = wavelet_denoise_1d(g[c].to_numpy())
            except Exception:
                pass
        out.append(g)
    return pd.concat(out, axis=0, ignore_index=True)

def make_sequences(df, feature_cols, target_col, seq_len):
    X_list, y_list = [], []
    for _, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        feats = g[feature_cols].to_numpy()
        tgt = g[target_col].to_numpy()
        if len(g) <= seq_len:
            continue
        for i in range(seq_len, len(g)):
            X_list.append(feats[i - seq_len:i, :])
            y_list.append(tgt[i])
    if not X_list:
        return np.empty((0, seq_len, len(feature_cols)), np.float32), np.empty((0,), np.float32)
    return np.stack(X_list), np.asarray(y_list)

@dataclass
class Metrics:
    rmse: float
    winrate: float

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, yh = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        ys.append(yb.cpu().numpy()); yh.append(pred.cpu().numpy())
    if not ys:
        return Metrics(float("nan"), float("nan"))
    y, yhat = np.concatenate(ys), np.concatenate(yh)
    return Metrics(math.sqrt(mean_squared_error(y, yhat)),
                   np.mean((y * yhat) > 0))

# ------------------------ Windows ------------------------

def iter_eval_windows_expanding(df, initial_years, val_months, test_months, step_months):
    """
    Expanding setup: first eval_start is after `initial_years`,
    then step forward by `step_months`.
    Yields (eval_start, val_end, test_end).
    """
    min_y = month_floor(df["y_date"].min())
    max_y = month_floor(df["y_date"].max())
    spans = []
    cur = min_y + relativedelta(years=int(initial_years))
    while True:
        eval_start = cur
        val_end    = eval_start + relativedelta(months=val_months)
        test_end   = val_end    + relativedelta(months=test_months)
        if eval_start >= max_y or test_end > max_y + relativedelta(months=1):
            break
        spans.append((eval_start, val_end, test_end))
        cur += relativedelta(months=step_months)
    return spans

def get_train_val_test_split_expanding(df: pd.DataFrame,
                                       eval_start: pd.Timestamp,
                                       val_end: pd.Timestamp,
                                       test_end: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # TRAIN = all before eval_start (expanding)
    train_mask = (df["y_date"] < eval_start)
    val_mask   = (df["y_date"] >= eval_start) & (df["y_date"] < val_end)
    test_mask  = (df["y_date"] >= val_end)    & (df["y_date"] < test_end)
    return df.loc[train_mask].copy(), df.loc[val_mask].copy(), df.loc[test_mask].copy()

# ------------------------ Training ------------------------

def train_one_window(train_df, val_df, test_df, feature_pool, target_col, seq_len,
                     window_tag, mdl_cfg):
    # FS slice from TRAIN
    fs_df = train_df
    if FS_LOOKBACK_MONTHS is not None and FS_LOOKBACK_MONTHS > 0:
        fs_start = val_df["y_date"].min() - relativedelta(months=FS_LOOKBACK_MONTHS)
        fs_df = train_df[train_df["y_date"] >= fs_start].copy()
        if len(fs_df) < 1000:
            fs_df = train_df

    selected = lgbm_feature_select(fs_df, feature_pool, target_col, mdl_cfg["TOP_K"], backend=mdl_cfg["FS_BACKEND"])
    print(f"[{window_tag}] Selected {len(selected)} feats (top-{mdl_cfg['TOP_K']}):")
    print("   " + ", ".join(selected[:25]) + (" ..." if len(selected) > 25 else ""))

    # Optional denoise
    train_df = denoise_split(train_df, selected)
    val_df   = denoise_split(val_df, selected)
    test_df  = denoise_split(test_df, selected)

    # Scale with TRAIN stats only
    scaler = StandardScaler()
    scaler.fit(train_df[selected].to_numpy(np.float32))
    for df_ in (train_df, val_df, test_df):
        df_.loc[:, selected] = scaler.transform(df_[selected].to_numpy(np.float32))

    # Seqs
    Xtr, ytr = make_sequences(train_df, selected, target_col, seq_len)
    Xva, yva = make_sequences(val_df,   selected, target_col, seq_len)
    Xte, yte = make_sequences(test_df,  selected, target_col, seq_len)
    if not len(Xtr) or not len(Xva):
        print(f"[{window_tag}] Skipped (not enough sequences).")
        return {"best_val_rmse": float("nan"), "best_val_winrate": float("nan")}

    tr_loader = DataLoader(SeqDataset(Xtr, ytr), batch_size=mdl_cfg["BATCH_SIZE"], shuffle=True)
    va_loader = DataLoader(SeqDataset(Xva, yva), batch_size=mdl_cfg["BATCH_SIZE"])
    te_loader = DataLoader(SeqDataset(Xte, yte), batch_size=mdl_cfg["BATCH_SIZE"]) if len(Xte) else None

    # Model + optimizer + YOUR GMADL
    model = XLSTMRegressor(len(selected), mdl_cfg["D_MODEL"], mdl_cfg["LAYERS"], mdl_cfg["DROPOUT"]).to(DEVICE)
    opt   = optim.AdamW(model.parameters(), lr=mdl_cfg["LR"], weight_decay=mdl_cfg["WEIGHT_DECAY"])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(mdl_cfg["EPOCHS"], 1))
    loss_fn = GMADL_Torch(lam=GMADL_LAMBDA, margin=GMADL_MARGIN, sig_k=SIG_K, abs_eps=ABS_EPS)

    best_rmse, stale = float("inf"), 0
    best_state = None

    for ep in range(1, mdl_cfg["EPOCHS"] + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        tr = evaluate(model, tr_loader, DEVICE)
        va = evaluate(model, va_loader, DEVICE)
        print(f"[{window_tag}] Epoch {ep:02d} | train RMSE={tr.rmse:.5f} win={tr.winrate:.3f} | "
              f"val RMSE={va.rmse:.5f} win={va.winrate:.3f}")

        if va.rmse < best_rmse:
            best_rmse, stale = va.rmse, 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= mdl_cfg["PATIENCE"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics
    va_best = evaluate(model, va_loader, DEVICE)
    if te_loader is not None:
        te_best = evaluate(model, te_loader, DEVICE)
        print(f"[{window_tag}] TEST  RMSE={te_best.rmse:.5f} win={te_best.winrate:.3f}")
    else:
        print(f"[{window_tag}] TEST  (no sequences)")

    return {"best_val_rmse": va_best.rmse, "best_val_winrate": va_best.winrate}

# ------------------------ Main (per-cluster) ------------------------

def main():
    # Load features
    df = pd.read_parquet(PARQUET_PATH)
    df = normalize_columns_df(df)
    df["date"] = pd.to_datetime(df["date"])

    # Merge clusters
    clusters = pd.read_csv(CLUSTERS_CSV)
    clusters.columns = [c.strip().lower() for c in clusters.columns]
    if not {"ticker","cluster"}.issubset(clusters.columns):
        raise ValueError("Clusters CSV must have columns: ticker, cluster")
    df = df.merge(clusters[["ticker","cluster"]], on="ticker", how="inner")

    # y_date
    df = build_y_date(df)

    # Feature pool (global numeric except identifiers/targets)
    exclude = {"ticker", "date", "y_date", "y_ret_next", "cluster"}
    base_pool = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    print(f"Global candidate features: {len(base_pool)}")

    # Per-cluster
    for clus, dfc in df.groupby("cluster", sort=True):
        dfc = dfc.copy()

        # per-cluster overrides
        cfg = {
            "SEQ_LEN": SEQ_LEN, "TOP_K": TOP_K, "BATCH_SIZE": BATCH_SIZE, "EPOCHS": EPOCHS,
            "PATIENCE": PATIENCE, "LR": LR, "WEIGHT_DECAY": WEIGHT_DECAY,
            "D_MODEL": D_MODEL, "LAYERS": LAYERS, "DROPOUT": DROPOUT, "FS_BACKEND": FS_BACKEND
        }
        if clus in CLUSTER_CONFIG:
            cfg.update(CLUSTER_CONFIG[clus])

        print(f"\n========== Cluster {clus} | rows={len(dfc):,} | tickers={dfc['ticker'].nunique()} ==========")
        print("Overrides:", {k: v for k, v in CLUSTER_CONFIG.get(clus, {}).items()})

        feature_pool = [c for c in base_pool if c in dfc.columns]
        print(f"Candidate features (cluster {clus}): {len(feature_pool)}")

        # Windows (expanding)
        spans = iter_eval_windows_expanding(dfc, INITIAL_TRAIN_YEARS, VAL_MONTHS, TEST_MONTHS, STEP_MONTHS)
        print(f"Total evaluation windows (cluster {clus}): {len(spans)} | initial_train_years={INITIAL_TRAIN_YEARS} (expanding)")

        results = []
        for i, (eval_start, val_end, test_end) in enumerate(spans, 1):
            train_df, val_df, test_df = get_train_val_test_split_expanding(dfc, eval_start, val_end, test_end)

            # ensure enough history for sequences
            if train_df.groupby("ticker").size().max() < cfg["SEQ_LEN"] + 1:
                print(f"[C{clus} W{i}] Skipping: not enough history for seq_len={cfg['SEQ_LEN']} in expanding train.")
                continue

            tag = (f"C{clus} W{i} init:{INITIAL_TRAIN_YEARS}y expanding | "
                   f"train:(-inf)→{eval_start.date()} | val:{eval_start.date()}→{val_end.date()} | "
                   f"test:{val_end.date()}→{test_end.date()}")
            print("===", tag, f"(train rows={len(train_df):,}, val rows={len(val_df):,}, test rows={len(test_df):,})", "===")

            res = train_one_window(train_df, val_df, test_df, feature_pool, "y_ret_next", cfg["SEQ_LEN"], tag, cfg)
            results.append({
                "window": i,
                "lookback_years": -1,              # expanding (kept column to preserve schema)
                "train_until": eval_start,
                "val_start": eval_start,
                "val_end": val_end,
                "best_val_rmse": res["best_val_rmse"],
                "best_val_winrate": res["best_val_winrate"],
            })

        out_csv = f"{OUTPUT_PREFIX}_cluster{clus}.csv"
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print(f"[Cluster {clus}] Saved results to {out_csv}")

if __name__ == "__main__":
    main()
