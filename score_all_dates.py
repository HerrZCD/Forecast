"""
score_all_dates.py
==================
加载训练好的 PortfolioMASTER 模型，为每个交易日打分，
并将原始分数与百分位排名分数同时写入 DuckDB 表 `tft_alpha_scores`。
"""

import math
import time as _time
import gc  # 导入垃圾回收模块
from datetime import date as _date
import duckdb
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

# 确保从你的主程序中导入相同的配置
from portfolio_master import (
    PortfolioMASTER, CrossSectionalDataset,
    DB_PATH, SEQ_LEN, FORWARD_DAYS, NUM_FEATURES, NUM_FEATURE_COLS,
    MACRO_COLS, MACRO_DIM, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT,
    SECTOR_EMBED_DIM, DEVICE, load_and_prepare_data,
)

MODEL_PATH = "./portfolio_master_best.pt"


def _ensure_scores_table(con):
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS tft_alpha_scores (
            date       DATE,
            ticker     VARCHAR,
            raw_score  DOUBLE,
            rank_score DOUBLE
        )
        """
    )


def _load_model(num_sectors):
    model = PortfolioMASTER(
        num_features=NUM_FEATURES, num_sectors=num_sectors, macro_dim=MACRO_DIM,
        d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, dropout=DROPOUT,
        n_layers=N_LAYERS, sector_embed_dim=SECTOR_EMBED_DIM, seq_len=SEQ_LEN,
    ).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        print(f"Model successfully loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: {MODEL_PATH} not found.")
        return None

    model.eval()
    return model


def _find_latest_eligible_di(all_dates, macro_array, univ_mask, target_date):
    target_ts = np.datetime64(target_date)
    eligible_di = None

    for di in range(len(all_dates) - 1, -1, -1):
        current_date = np.datetime64(all_dates[di])
        if current_date > target_ts:
            continue
        if di < SEQ_LEN - 1:
            continue
        if np.isnan(macro_array[di]).any():
            continue
        if univ_mask[di].sum() < 5:
            continue
        eligible_di = di
        break

    return eligible_di


def _score_and_save_one_date(
    model,
    feat_array,
    sector_array,
    target_array,
    univ_mask,
    macro_array,
    all_dates,
    all_symbols,
    di,
):
    target_date = pd.to_datetime(all_dates[di]).date()

    ds = CrossSectionalDataset(
        feat_array, sector_array, target_array, univ_mask, macro_array, [di]
    )
    sample = ds[0]
    if sample is None:
        print(f"Skip: sample is None on {target_date}.")
        return 0

    with torch.no_grad():
        x_num, x_cat, m_vec, tgt, msk = [s.to(DEVICE) for s in sample]
        logits = model(x_num, x_cat, m_vec)

    raw_scores = logits.cpu().numpy()
    norm_ranks = rankdata(raw_scores) / len(raw_scores)
    stock_ids = np.where(univ_mask[di])[0]

    records = []
    for j, sid in enumerate(stock_ids):
        if j < len(raw_scores):
            records.append(
                {
                    "date": all_dates[di],
                    "ticker": all_symbols[sid],
                    "raw_score": float(raw_scores[j]),
                    "rank_score": float(norm_ranks[j]),
                }
            )

    if not records:
        print(f"Skip: no records generated on {target_date}.")
        return 0

    df_day = pd.DataFrame(records)
    df_day["date"] = pd.to_datetime(df_day["date"]).dt.date

    con = duckdb.connect(DB_PATH)
    con.execute("SET memory_limit='256MB'")
    _ensure_scores_table(con)
    con.execute("DELETE FROM tft_alpha_scores WHERE date = ?", [target_date])
    con.register("df_day", df_day)
    con.execute(
        """
        INSERT INTO tft_alpha_scores (date, ticker, raw_score, rank_score)
        SELECT date, ticker, raw_score, rank_score FROM df_day
        """
    )
    inserted = con.execute(
        "SELECT COUNT(*) FROM tft_alpha_scores WHERE date = ?", [target_date]
    ).fetchone()[0]
    con.close()

    print(f"SUCCESS: {target_date} ({inserted} rows)")
    return inserted


def update_latest_date_score(run_date=None):
    """
    仅更新指定日期（默认今天）的分数。
    """
    target_date = pd.to_datetime(run_date).date() if run_date is not None else _date.today()

    print("=" * 60)
    print(f"Incremental Scoring: target_date={target_date}")
    print("=" * 60)

    # 1. 加载数据
    (feat_array, sector_array, target_array, univ_mask, macro_array,
     all_dates, all_symbols, num_sectors, date2idx) = load_and_prepare_data()

    # 2. 判断目标日期是否有数据
    di = date2idx.get(np.datetime64(target_date))
    if di is None:
        print(f"Skip: {target_date} not found in market data calendar.")
        return 0

    # 3. 判断是否满足打分条件
    if di < SEQ_LEN - 1:
        print(f"Skip: {target_date} does not have enough history for seq_len={SEQ_LEN}.")
        return 0
    if np.isnan(macro_array[di]).any():
        print(f"Skip: macro data is missing on {target_date}.")
        return 0
    if univ_mask[di].sum() < 5:
        print(f"Skip: insufficient universe size on {target_date}.")
        return 0

    # 4. 加载模型
    model = _load_model(num_sectors)
    if model is None:
        return 0

    return _score_and_save_one_date(
        model,
        feat_array,
        sector_array,
        target_array,
        univ_mask,
        macro_array,
        all_dates,
        all_symbols,
        di,
    )


def ensure_latest_available_score(run_date=None):
    """
    确保截至指定日期（默认今天），最近一个可打分交易日已经有 score。
    若今天非交易日，则自动回退到最近一个交易日。
    若库里该日期没有 score，则补算；有则跳过。
    """
    target_date = pd.to_datetime(run_date).date() if run_date is not None else _date.today()

    print("=" * 60)
    print(f"Daily Sync: target_date={target_date}")
    print("=" * 60)

    (feat_array, sector_array, target_array, univ_mask, macro_array,
     all_dates, all_symbols, num_sectors, date2idx) = load_and_prepare_data()

    di = _find_latest_eligible_di(all_dates, macro_array, univ_mask, target_date)
    if di is None:
        print(f"Skip: no eligible trading day found on or before {target_date}.")
        return 0

    score_date = pd.to_datetime(all_dates[di]).date()
    if score_date != target_date:
        print(f"Resolved latest eligible trading day: {score_date}")

    con = duckdb.connect(DB_PATH)
    con.execute("SET memory_limit='256MB'")
    _ensure_scores_table(con)
    existing = con.execute(
        "SELECT COUNT(*) FROM tft_alpha_scores WHERE date = ?", [score_date]
    ).fetchone()[0]
    con.close()

    if existing > 0:
        print(f"Skip: {score_date} already has {existing} score rows.")
        return existing

    model = _load_model(num_sectors)
    if model is None:
        return 0

    return _score_and_save_one_date(
        model,
        feat_array,
        sector_array,
        target_array,
        univ_mask,
        macro_array,
        all_dates,
        all_symbols,
        di,
    )


def main():
    print("=" * 60)
    print("Scoring System: PortfolioMASTER Daily Inference (Memory Optimized)")
    print("=" * 60)

    # 1. 加载数据
    (feat_array, sector_array, target_array, univ_mask, macro_array,
     all_dates, all_symbols, num_sectors, date2idx) = load_and_prepare_data()

    valid_dis = []
    for di in range(SEQ_LEN - 1, len(all_dates)):
        if not np.isnan(macro_array[di]).any() and univ_mask[di].sum() >= 5:
            valid_dis.append(di)

    print(f"Total valid trading days: {len(valid_dis)}")
    all_ds = CrossSectionalDataset(
        feat_array, sector_array, target_array, univ_mask, macro_array, valid_dis
    )

    # 2. 初始化模型
    model = PortfolioMASTER(
        num_features=NUM_FEATURES, num_sectors=num_sectors, macro_dim=MACRO_DIM,
        d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, dropout=DROPOUT,
        n_layers=N_LAYERS, sector_embed_dim=SECTOR_EMBED_DIM, seq_len=SEQ_LEN,
    ).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        print(f"Model successfully loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: {MODEL_PATH} not found.")
        return

    model.eval()

    # 3. 初始化数据库
    con = duckdb.connect(DB_PATH)
    con.execute("SET memory_limit='256MB'") # 严格限制 DuckDB 内存
    con.execute("DROP TABLE IF EXISTS tft_alpha_scores")
    _ensure_scores_table(con)

    # 4. 执行截面打分 (分批次处理)
    print("Starting inference...")
    t0 = _time.time()
    
    temp_records = []
    batch_size = 50  # 每 50 天写入一次硬盘，释放内存

    with torch.no_grad():
        for i in range(len(all_ds)):
            sample = all_ds[i]
            if sample is None: continue
            
            x_num, x_cat, m_vec, tgt, msk = [s.to(DEVICE) for s in sample]
            logits = model(x_num, x_cat, m_vec)
            
            raw_scores = logits.cpu().numpy()
            norm_ranks = rankdata(raw_scores) / len(raw_scores)

            di = all_ds.date_indices[i]
            current_date = all_dates[di]
            stock_ids = np.where(univ_mask[di])[0]

            for j, sid in enumerate(stock_ids):
                if j < len(raw_scores):
                    # 使用元组存储，比字典更节省内存
                    temp_records.append((
                        current_date, 
                        all_symbols[sid], 
                        float(raw_scores[j]), 
                        float(norm_ranks[j])
                    ))

            # --- 定期“卸货”到 DuckDB ---
            if (i + 1) % batch_size == 0 or (i + 1) == len(all_ds):
                df_batch = pd.DataFrame(temp_records, columns=['date', 'ticker', 'raw_score', 'rank_score'])
                df_batch["date"] = pd.to_datetime(df_batch["date"]).dt.date
                
                con.register("df_batch_tmp", df_batch)
                con.execute("INSERT INTO tft_alpha_scores SELECT * FROM df_batch_tmp")
                
                print(f"  Processed {i+1}/{len(all_ds)} days... (Batch saved, memory released)")
                
                # 彻底销毁临时对象并回收内存
                temp_records = []
                del df_batch
                gc.collect()

    # 5. 验证结果
    final_count = con.execute("SELECT COUNT(*) FROM tft_alpha_scores").fetchone()[0]
    sample_data = con.execute("SELECT * FROM tft_alpha_scores ORDER BY date DESC, rank_score DESC LIMIT 5").fetchdf()
    con.close()

    print("-" * 60)
    print(f"SUCCESS! Total Records: {final_count:,}")
    print(f"Inference Time: {_time.time()-t0:.1f}s")
    print("\nTop 5 ranked stocks from the latest date:")
    print(sample_data)
    print("-" * 60)

if __name__ == "__main__":
    main()