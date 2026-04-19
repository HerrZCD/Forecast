"""
PortfolioMASTER: Market-Guided Spatio-Temporal Transformer for Cross-Sectional Stock Ranking
============================================================================================
Predicts 5-day forward cross-sectional stock return rankings in the S&P 500 universe.
Optimized vectorized data pipeline for fast training.
"""

import math
import time as _time
import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

# ============================================================
# 0. Configuration
# ============================================================
DB_PATH = "/home/zhangchundong/DailySync/data_loader//stock_prices.duckdb"
SEQ_LEN = 20          # lookback window (trading days)
FORWARD_DAYS = 5      # prediction horizon
D_MODEL = 64           # transformer hidden dim
N_HEADS = 4
N_LAYERS = 2
D_FF = 128
DROPOUT = 0.1
SECTOR_EMBED_DIM = 8
LR = 1e-4
WEIGHT_DECAY = 1e-3
EPOCHS = 30
TRAIN_RATIO = 0.8
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")

NUM_FEATURE_COLS = [
    "open", "high", "low", "close", "vwap",
    "chg", "pct_chg", "volume", "amt", "turn",
    "sentiment_normalized", "sentiment_count",
]
MACRO_COLS = ["vix_close", "sp_ret_1d", "sp_ret_5d", "sp_ret_20d",
              "sp_bias_20", "sp_bias_60", "sp_hvol_20"]
MACRO_DIM = len(MACRO_COLS)
NUM_FEATURES = len(NUM_FEATURE_COLS)

print(f"Device: {DEVICE}")


# ============================================================
# 1. Data Loading & Feature Engineering (Vectorized)
# ============================================================
def load_and_prepare_data():
    t0 = _time.time()
    con = duckdb.connect(DB_PATH, read_only=True)

    # ---- Load tables ----
    print("Loading tables from DuckDB...")
    daily_prices = con.execute("""
        SELECT thscode AS symbol,
               CAST(time AS DATE) AS date,
               open, high, low, close, vwap, chg, pct_chg, volume, amt, turn
        FROM daily_prices ORDER BY thscode, time
    """).fetchdf()

    sp500_daily = con.execute("SELECT date, close AS sp_close FROM sp500_daily ORDER BY date").fetchdf()
    vix_daily = con.execute("SELECT date, close AS vix_close FROM vix_daily ORDER BY date").fetchdf()
    sp500_universe = con.execute("SELECT date AS univ_date, symbol FROM sp500_universe ORDER BY date, symbol").fetchdf()
    sentiment = con.execute("""
        SELECT ticker, CAST(date AS DATE) AS date,
               normalized AS sentiment_normalized, count AS sentiment_count
        FROM us_stock_sentiment
    """).fetchdf()
    companies = con.execute("SELECT ticker, sector FROM companies WHERE sector IS NOT NULL").fetchdf()
    con.close()
    print(f"  Tables loaded in {_time.time()-t0:.1f}s")

    # ---- Sector encoding ----
    sectors = sorted(companies["sector"].unique().tolist())
    sector_map = {s: i for i, s in enumerate(sectors)}
    num_sectors = len(sectors)
    ticker_sector = dict(zip(companies["ticker"], companies["sector"].map(sector_map)))
    print(f"  Sectors ({num_sectors}): {sectors}")

    # ---- Macro state vector ----
    sp = sp500_daily.sort_values("date").reset_index(drop=True)
    sp["sp_ret_1d"] = sp["sp_close"].pct_change(1)
    sp["sp_ret_5d"] = sp["sp_close"].pct_change(5)
    sp["sp_ret_20d"] = sp["sp_close"].pct_change(20)
    sp["sp_ma20"] = sp["sp_close"].rolling(20).mean()
    sp["sp_ma60"] = sp["sp_close"].rolling(60).mean()
    sp["sp_bias_20"] = (sp["sp_close"] - sp["sp_ma20"]) / sp["sp_ma20"]
    sp["sp_bias_60"] = (sp["sp_close"] - sp["sp_ma60"]) / sp["sp_ma60"]
    sp["sp_hvol_20"] = sp["sp_ret_1d"].rolling(20).std() * math.sqrt(252)

    macro = pd.merge(sp[["date"] + MACRO_COLS[1:]], vix_daily[["date", "vix_close"]], on="date", how="inner")
    macro = macro.dropna(subset=MACRO_COLS).reset_index(drop=True)
    macro_dict = macro.set_index("date")[MACRO_COLS].to_dict("index")
    print(f"  Macro data: {len(macro)} days ({macro['date'].min()} ~ {macro['date'].max()})")

    # ---- Forward-fill universe to daily ----
    univ_dates = sorted(sp500_universe["univ_date"].unique())
    univ_dict = sp500_universe.groupby("univ_date")["symbol"].apply(set).to_dict()

    # ---- Merge sentiment ----
    daily_prices["bare_ticker"] = daily_prices["symbol"].str.split(".").str[0]
    sentiment = sentiment.drop_duplicates(subset=["ticker", "date"], keep="last")
    daily_prices = daily_prices.merge(
        sentiment, left_on=["bare_ticker", "date"], right_on=["ticker", "date"], how="left"
    )
    daily_prices["sentiment_normalized"] = daily_prices["sentiment_normalized"].fillna(0.0)
    daily_prices["sentiment_count"] = daily_prices["sentiment_count"].fillna(0).astype(float)
    daily_prices["sector_id"] = daily_prices["bare_ticker"].map(ticker_sector).fillna(0).astype(int)

    # ---- Forward return ----
    daily_prices = daily_prices.sort_values(["symbol", "date"]).reset_index(drop=True)
    daily_prices["fwd_5d_ret"] = (
        daily_prices.groupby("symbol")["close"].shift(-FORWARD_DAYS) / daily_prices["close"] - 1
    )

    # ---- Build pivot tables for fast vectorized access ----
    print("Building pivot tables (this may take a moment)...")
    t1 = _time.time()

    all_symbols = sorted(daily_prices["symbol"].unique())
    all_dates = sorted(daily_prices["date"].unique())
    sym2idx = {s: i for i, s in enumerate(all_symbols)}
    date2idx = {d: i for i, d in enumerate(all_dates)}
    N_sym = len(all_symbols)
    N_dates = len(all_dates)

    # Pre-allocate arrays: (N_dates, N_sym, N_features)
    feat_array = np.full((N_dates, N_sym, NUM_FEATURES), np.nan, dtype=np.float32)
    sector_array = np.zeros((N_sym,), dtype=np.int64)
    target_array = np.full((N_dates, N_sym), np.nan, dtype=np.float32)

    # Fill sector array
    for sym in all_symbols:
        bare = sym.split(".")[0]
        sector_array[sym2idx[sym]] = ticker_sector.get(bare, 0)

    # Fill feature and target arrays via vectorized pivot
    daily_prices["sym_idx"] = daily_prices["symbol"].map(sym2idx)
    daily_prices["date_idx"] = daily_prices["date"].map(date2idx)

    sym_idxs = daily_prices["sym_idx"].values
    date_idxs = daily_prices["date_idx"].values
    feat_vals = daily_prices[NUM_FEATURE_COLS].values.astype(np.float32)
    tgt_vals = daily_prices["fwd_5d_ret"].values.astype(np.float32)

    feat_array[date_idxs, sym_idxs, :] = feat_vals
    target_array[date_idxs, sym_idxs] = tgt_vals

    # Forward-fill NaN price features along time axis per stock
    for fi in range(NUM_FEATURES):
        col = feat_array[:, :, fi]  # (N_dates, N_sym)
        df_col = pd.DataFrame(col)
        feat_array[:, :, fi] = df_col.ffill().values

    # Replace remaining NaN with 0
    feat_array = np.nan_to_num(feat_array, nan=0.0)

    print(f"  Pivot tables built in {_time.time()-t1:.1f}s")
    print(f"  Shape: dates={N_dates}, symbols={N_sym}, features={NUM_FEATURES}")

    # ---- Build daily universe masks ----
    univ_mask = np.zeros((N_dates, N_sym), dtype=bool)
    ui = 0
    for di, d in enumerate(all_dates):
        while ui < len(univ_dates) - 1 and univ_dates[ui + 1] <= d:
            ui += 1
        if ui < len(univ_dates) and univ_dates[ui] <= d:
            members = univ_dict[univ_dates[ui]]
            for s in members:
                if s in sym2idx:
                    univ_mask[di, sym2idx[s]] = True

    # ---- Macro array ----
    macro_array = np.full((N_dates, MACRO_DIM), np.nan, dtype=np.float32)
    for di, d in enumerate(all_dates):
        if d in macro_dict:
            macro_array[di] = [macro_dict[d][c] for c in MACRO_COLS]

    # Forward-fill macro
    macro_df = pd.DataFrame(macro_array)
    macro_array = macro_df.ffill().bfill().values.astype(np.float32)

    print(f"Total data preparation: {_time.time()-t0:.1f}s")

    return (feat_array, sector_array, target_array, univ_mask, macro_array,
            all_dates, all_symbols, num_sectors, date2idx)


# ============================================================
# 2. Dataset (Pre-materialized, Date-wise)
# ============================================================
class CrossSectionalDataset:
    """
    Each sample = one trading day cross-section.
    Pre-materialized for speed; __getitem__ does Z-score normalization on the fly.
    """

    def __init__(self, feat_array, sector_array, target_array, univ_mask,
                 macro_array, date_indices):
        self.feat = feat_array
        self.sector = sector_array
        self.target = target_array
        self.univ_mask = univ_mask
        self.macro = macro_array
        self.date_indices = date_indices

    def __len__(self):
        return len(self.date_indices)

    def __getitem__(self, idx):
        di = self.date_indices[idx]

        # Universe stocks for this date
        mask = self.univ_mask[di]
        stock_ids = np.where(mask)[0]
        if len(stock_ids) < 10:
            return None

        start_di = di - SEQ_LEN + 1
        if start_di < 0:
            return None

        # X_num: (N_stocks, SEQ_LEN, F)
        x_num = self.feat[start_di:di + 1, stock_ids, :]  # (SEQ_LEN, N_stocks, F)
        x_num = x_num.transpose(1, 0, 2)  # (N_stocks, SEQ_LEN, F)

        # Cross-sectional Z-score: normalize across stocks for each (time, feature)
        mean = np.mean(x_num, axis=0, keepdims=True)
        std = np.std(x_num, axis=0, keepdims=True) + 1e-8
        x_num = (x_num - mean) / std

        # X_cat: sector ids
        x_cat = self.sector[stock_ids]

        # Macro
        m_vec = self.macro[di]

        # Target & validity mask
        targets = self.target[di, stock_ids]
        valid_mask = (~np.isnan(targets)).astype(np.float32)
        targets = np.nan_to_num(targets, nan=0.0)

        return (
            torch.from_numpy(x_num),
            torch.from_numpy(x_cat.copy()),
            torch.from_numpy(m_vec.copy()),
            torch.from_numpy(targets),
            torch.from_numpy(valid_mask),
        )


def build_datasets(feat_array, sector_array, target_array, univ_mask,
                   macro_array, all_dates, num_sectors, date2idx):
    """Split into train/val by date."""
    valid_dis = []
    for di in range(SEQ_LEN - 1, len(all_dates) - FORWARD_DAYS):
        if np.isnan(macro_array[di]).any():
            continue
        if univ_mask[di].sum() < 10:
            continue
        valid_dis.append(di)

    # OOS split: last 1 year (~252 trading days) for validation
    from datetime import timedelta
    last_date = all_dates[valid_dis[-1]]
    oos_start = last_date - timedelta(days=365)
    train_dis = [di for di in valid_dis if all_dates[di] < oos_start]
    val_dis = [di for di in valid_dis if all_dates[di] >= oos_start]

    print(f"Valid dates: {len(valid_dis)} | Train: {len(train_dis)} | Val(OOS 1Y): {len(val_dis)}")
    if train_dis:
        print(f"Train: {all_dates[train_dis[0]]} ~ {all_dates[train_dis[-1]]}")
    if val_dis:
        print(f"OOS:   {all_dates[val_dis[0]]} ~ {all_dates[val_dis[-1]]}")

    train_ds = CrossSectionalDataset(feat_array, sector_array, target_array,
                                     univ_mask, macro_array, train_dis)
    val_ds = CrossSectionalDataset(feat_array, sector_array, target_array,
                                   univ_mask, macro_array, val_dis)
    return train_ds, val_ds


# ============================================================
# 3. Model: PortfolioMASTER
# ============================================================
class MarketGuidedGating(nn.Module):
    """Macro vector -> per-feature scaling coefficients via Softmax."""

    def __init__(self, macro_dim, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(macro_dim, d_model),
            nn.Softmax(dim=-1),
        )
        self.d_model = d_model

    def forward(self, m, x):
        """
        m: (macro_dim,) or (1, macro_dim)
        x: (N, T, D)
        """
        if m.dim() == 1:
            m = m.unsqueeze(0)
        g = self.gate(m) * self.d_model  # (1, D), mean ~ 1
        return x * g.unsqueeze(1)  # broadcast over N and T


class IntraStockTemporalBlock(nn.Module):
    """Per-stock temporal self-attention."""

    def __init__(self, d_model, n_heads, d_ff, dropout, n_layers):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x):
        return self.encoder(x)


class InterStockSpatialBlock(nn.Module):
    """Cross-sectional attention: stocks attend to each other per time step."""

    def __init__(self, d_model, n_heads, d_ff, dropout, n_layers):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x):
        # x: (N, T, D) -> (T, N, D)
        x = x.permute(1, 0, 2)
        x = self.encoder(x)
        x = x.permute(1, 0, 2)
        return x


class PortfolioMASTER(nn.Module):
    """
    Architecture:
      1. Feature Fusion: sector embedding + numerical features -> d_model
      2. Market-Guided Gating: macro vector gates features
      3. Intra-Stock Temporal Attention: per-stock time-series attention
      4. Inter-Stock Spatial Attention: cross-sectional attention
      5. Scorer: temporal pooling -> FC -> scalar logit per stock
    """

    def __init__(self, num_features, num_sectors, macro_dim, d_model, n_heads,
                 d_ff, dropout, n_layers, sector_embed_dim, seq_len):
        super().__init__()
        self.sector_embed = nn.Embedding(num_sectors + 1, sector_embed_dim)
        input_dim = num_features + sector_embed_dim
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        self.market_gate = MarketGuidedGating(macro_dim, d_model)
        self.temporal = IntraStockTemporalBlock(d_model, n_heads, d_ff, dropout, n_layers)
        self.spatial = InterStockSpatialBlock(d_model, n_heads, d_ff, dropout, n_layers)

        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x_num, x_cat, m):
        """
        x_num: (N, T, F_num)
        x_cat: (N,)
        m:     (MACRO_DIM,) or (1, MACRO_DIM)
        Returns: (N,) logits
        """
        N, T, _ = x_num.shape
        sec = self.sector_embed(x_cat).unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([x_num, sec], dim=-1)
        x = self.input_proj(x)
        x = x + self.pos_enc[:, :T, :]

        x = self.market_gate(m, x)
        x = self.temporal(x)
        x = self.spatial(x)

        x = x.mean(dim=1)  # temporal pooling
        return self.scorer(x).squeeze(-1)


# ============================================================
# 4. ListNet Ranking Loss
# ============================================================
class ListNetLoss(nn.Module):
    """
    ListNet top-1 probability loss.
    Softmax on both predicted scores and true returns (cross-sectional),
    then cross-entropy.
    """

    def forward(self, logits, targets, mask):
        valid = mask.bool()
        if valid.sum() < 5:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        pred = logits[valid]
        true = targets[valid]
        p_pred = F.log_softmax(pred, dim=0)
        p_true = F.softmax(true, dim=0)
        return -torch.sum(p_true * p_pred)


# ============================================================
# 5. Training & Evaluation
# ============================================================
def rank_ic(logits, targets, mask):
    valid = mask.bool().cpu().numpy()
    if valid.sum() < 5:
        return float("nan")
    p = logits.detach().cpu().numpy()[valid]
    t = targets.detach().cpu().numpy()[valid]
    ic, _ = spearmanr(p, t)
    return ic


def train_epoch(model, ds, criterion, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for i in range(len(ds)):
        sample = ds[i]
        if sample is None:
            continue
        x_num, x_cat, m, tgt, msk = [s.to(device) for s in sample]
        optimizer.zero_grad()
        logits = model(x_num, x_cat, m)
        loss = criterion(logits, tgt, msk)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_epoch(model, ds, criterion, device):
    model.eval()
    total_loss, ics, n = 0.0, [], 0
    for i in range(len(ds)):
        sample = ds[i]
        if sample is None:
            continue
        x_num, x_cat, m, tgt, msk = [s.to(device) for s in sample]
        logits = model(x_num, x_cat, m)
        loss = criterion(logits, tgt, msk)
        total_loss += loss.item()
        ic = rank_ic(logits, tgt, msk)
        if not np.isnan(ic):
            ics.append(ic)
        n += 1
    return total_loss / max(n, 1), np.mean(ics) if ics else float("nan")


# ============================================================
# 6. Main
# ============================================================
def main():
    print("=" * 60)
    print("PortfolioMASTER — Data Preparation")
    print("=" * 60)

    (feat_array, sector_array, target_array, univ_mask, macro_array,
     all_dates, all_symbols, num_sectors, date2idx) = load_and_prepare_data()

    train_ds, val_ds = build_datasets(
        feat_array, sector_array, target_array, univ_mask,
        macro_array, all_dates, num_sectors, date2idx
    )

    print(f"\nModel config: d={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS}, "
          f"ff={D_FF}, sectors={num_sectors}, seq={SEQ_LEN}")
    model = PortfolioMASTER(
        num_features=NUM_FEATURES, num_sectors=num_sectors, macro_dim=MACRO_DIM,
        d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, dropout=DROPOUT,
        n_layers=N_LAYERS, sector_embed_dim=SECTOR_EMBED_DIM, seq_len=SEQ_LEN,
    ).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = ListNetLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.1)

    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    best_ic = -1.0
    for ep in range(1, EPOCHS + 1):
        t0 = _time.time()
        tr_loss = train_epoch(model, train_ds, criterion, optimizer, DEVICE)
        vl_loss, vl_ic = eval_epoch(model, val_ds, criterion, DEVICE)
        scheduler.step()
        elapsed = _time.time() - t0

        tag = ""
        if not np.isnan(vl_ic) and vl_ic > best_ic:
            best_ic = vl_ic
            torch.save(model.state_dict(), "portfolio_master_best.pt")
            tag = " *"

        print(f"Epoch {ep:3d}/{EPOCHS}  "
              f"TrLoss={tr_loss:.4f}  VlLoss={vl_loss:.4f}  "
              f"RankIC={vl_ic:+.4f}  LR={optimizer.param_groups[0]['lr']:.2e}  "
              f"({elapsed:.1f}s){tag}")

    print("\n" + "=" * 60)
    print(f"Done. Best Val Rank IC = {best_ic:+.4f}")
    print("Model saved: portfolio_master_best.pt")
    print("=" * 60)

    # ==================================================================
    # 7. OOS Evaluation: Daily IC, ICIR, Long-Short Return
    # ==================================================================
    print("\n" + "=" * 60)
    print("OOS Evaluation (loading best model)")
    print("=" * 60)
    model.load_state_dict(torch.load("portfolio_master_best.pt", map_location=DEVICE, weights_only=True))
    model.eval()

    daily_ics = []
    daily_long_short = []
    oos_dates_used = []

    with torch.no_grad():
        for i in range(len(val_ds)):
            sample = val_ds[i]
            if sample is None:
                continue
            x_num, x_cat, m_vec, tgt, msk = [s.to(device=DEVICE) for s in sample]
            logits = model(x_num, x_cat, m_vec)

            valid = msk.bool()
            if valid.sum() < 20:
                continue

            pred_np = logits[valid].cpu().numpy()
            true_np = tgt[valid].cpu().numpy()

            ic, _ = spearmanr(pred_np, true_np)
            if np.isnan(ic):
                continue
            daily_ics.append(ic)

            # Long-short: top 20% - bottom 20%
            n_valid = len(pred_np)
            q = max(n_valid // 5, 1)
            sorted_idx = np.argsort(pred_np)
            long_ret = np.mean(true_np[sorted_idx[-q:]])
            short_ret = np.mean(true_np[sorted_idx[:q]])
            daily_long_short.append(long_ret - short_ret)

            di = val_ds.date_indices[i]
            oos_dates_used.append(str(all_dates[di]))

    daily_ics = np.array(daily_ics)
    daily_long_short = np.array(daily_long_short)

    mean_ic = np.mean(daily_ics)
    std_ic = np.std(daily_ics)
    icir = mean_ic / std_ic if std_ic > 1e-8 else float("nan")
    ic_pos_rate = (daily_ics > 0).mean() * 100
    mean_ls = np.mean(daily_long_short)
    std_ls = np.std(daily_long_short)
    ls_ir = mean_ls / std_ls if std_ls > 1e-8 else float("nan")

    print(f"\nOOS Period: {oos_dates_used[0]} ~ {oos_dates_used[-1]}  ({len(daily_ics)} days)")
    print("-" * 50)
    print(f"  Mean Rank IC :  {mean_ic:+.4f}")
    print(f"  Std  Rank IC :  {std_ic:.4f}")
    print(f"  ICIR         :  {icir:+.4f}")
    print(f"  IC > 0 Rate  :  {ic_pos_rate:.1f}%")
    print("-" * 50)
    print(f"  L/S Mean 5D  :  {mean_ls*100:+.2f}%")
    print(f"  L/S Std  5D  :  {std_ls*100:.2f}%")
    print(f"  L/S IR       :  {ls_ir:+.4f}")
    print("-" * 50)

    # Monthly breakdown
    print("\nMonthly Rank IC Breakdown:")
    print(f"  {'Month':>10s}  {'MeanIC':>8s}  {'ICIR':>8s}  {'#Days':>6s}")
    month_map = {}
    for dt_str, ic_val in zip(oos_dates_used, daily_ics):
        m = dt_str[:7]
        month_map.setdefault(m, []).append(ic_val)
    for m in sorted(month_map.keys()):
        arr = np.array(month_map[m])
        m_ic = np.mean(arr)
        m_ir = m_ic / np.std(arr) if np.std(arr) > 1e-8 else float("nan")
        print(f"  {m:>10s}  {m_ic:>+8.4f}  {m_ir:>+8.4f}  {len(arr):>6d}")

    print("\n" + "=" * 60)
    print("All done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
