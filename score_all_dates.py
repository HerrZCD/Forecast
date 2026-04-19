"""
score_all_dates.py
==================
加载训练好的 PortfolioMASTER 模型，为每个交易日打分，
并将原始分数与百分位排名分数同时写入 DuckDB 表 `tft_alpha_scores`。
"""

import math
import time as _time
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


def update_latest_date_score(run_date=None):
    """
    仅更新指定日期（默认今天）的分数。
    若该日期无量价数据（例如周末）或不满足打分条件，则自动跳过。
    """
    target_date = pd.to_datetime(run_date).date() if run_date is not None else _date.today()

    print("=" * 60)
    print(f"Incremental Scoring: target_date={target_date}")
    print("=" * 60)

    # 1. 加载数据
    (feat_array, sector_array, target_array, univ_mask, macro_array,
     all_dates, all_symbols, num_sectors, date2idx) = load_and_prepare_data()

    # 2. 判断目标日期是否有数据（周末/节假日通常不在 all_dates）
    di = date2idx.get(np.datetime64(target_date))
    if di is None:
        print(f"Skip: {target_date} not found in market data calendar.")
        return

    # 3. 判断是否满足打分条件
    if di < SEQ_LEN - 1:
        print(f"Skip: {target_date} does not have enough history for seq_len={SEQ_LEN}.")
        return
    if np.isnan(macro_array[di]).any():
        print(f"Skip: macro data is missing on {target_date}.")
        return
    if univ_mask[di].sum() < 5:
        print(f"Skip: insufficient universe size on {target_date}.")
        return

    # 4. 加载模型
    model = PortfolioMASTER(
        num_features=NUM_FEATURES, num_sectors=num_sectors, macro_dim=MACRO_DIM,
        d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, dropout=DROPOUT,
        n_layers=N_LAYERS, sector_embed_dim=SECTOR_EMBED_DIM, seq_len=SEQ_LEN,
    ).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        print(f"Model successfully loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: {MODEL_PATH} not found. Please train the model first.")
        return

    model.eval()

    # 5. 单日打分
    ds = CrossSectionalDataset(
        feat_array, sector_array, target_array, univ_mask, macro_array, [di]
    )
    sample = ds[0]
    if sample is None:
        print(f"Skip: sample is None on {target_date}.")
        return

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
        return

    # 6. 写入 DuckDB（同一天先删后插）
    df_day = pd.DataFrame(records)
    df_day["date"] = pd.to_datetime(df_day["date"]).dt.date

    con = duckdb.connect(DB_PATH)
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

    print("-" * 60)
    print(f"SUCCESS: updated {inserted} rows for {target_date}")
    print("-" * 60)

def main():
    print("=" * 60)
    print("Scoring System: PortfolioMASTER Daily Inference")
    print("=" * 60)

    # 1. 加载数据
    (feat_array, sector_array, target_array, univ_mask, macro_array,
     all_dates, all_symbols, num_sectors, date2idx) = load_and_prepare_data()

    # 2. 筛选可预测的日期 (必须有 SEQ_LEN 历史数据且宏观数据不为空)
    valid_dis = []
    for di in range(SEQ_LEN - 1, len(all_dates)):
        if np.isnan(macro_array[di]).any():
            continue
        if univ_mask[di].sum() < 5:  # 至少5只股票在池子里
            continue
        valid_dis.append(di)

    print(f"Total valid trading days: {len(valid_dis)}")
    all_ds = CrossSectionalDataset(
        feat_array, sector_array, target_array, univ_mask, macro_array, valid_dis
    )

    # 3. 加载最佳模型
    model = PortfolioMASTER(
        num_features=NUM_FEATURES, num_sectors=num_sectors, macro_dim=MACRO_DIM,
        d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, dropout=DROPOUT,
        n_layers=N_LAYERS, sector_embed_dim=SECTOR_EMBED_DIM, seq_len=SEQ_LEN,
    ).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        print(f"Model successfully loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: {MODEL_PATH} not found. Please train the model first.")
        return

    model.eval()

    # 4. 执行截面打分
    print("Starting inference...")
    t0 = _time.time()
    records = []

    with torch.no_grad():
        for i in range(len(all_ds)):
            sample = all_ds[i]
            if sample is None:
                continue
            
            # 准备张量并推断
            x_num, x_cat, m_vec, tgt, msk = [s.to(DEVICE) for s in sample]
            logits = model(x_num, x_cat, m_vec)
            
            # 将 Logits 转换为原始分数数组
            raw_scores = logits.cpu().numpy()
            
            # 关键改动：计算百分位排名 (0=最差, 1=最强)
            # rankdata 会给出 [1, 2, 3...] 排名，除以长度转为 [0.002, ... 1.0]
            norm_ranks = rankdata(raw_scores) / len(raw_scores)

            di = all_ds.date_indices[i]
            current_date = all_dates[di]
            
            # 获取当天 Universe 里的真实索引
            stock_ids = np.where(univ_mask[di])[0]

            for j, sid in enumerate(stock_ids):
                if j < len(raw_scores):
                    records.append({
                        "date": current_date,
                        "ticker": all_symbols[sid],
                        "raw_score": float(raw_scores[j]),
                        "rank_score": float(norm_ranks[j])
                    })

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(all_ds)} days...")

    # 5. 写入 DuckDB
    print("\nSaving results to DuckDB...")
    df_results = pd.DataFrame(records)
    df_results["date"] = pd.to_datetime(df_results["date"]).dt.date

    con = duckdb.connect(DB_PATH)
    con.execute("DROP TABLE IF EXISTS tft_alpha_scores")
    con.execute("""
        CREATE TABLE tft_alpha_scores (
            date       DATE,
            ticker     VARCHAR,
            raw_score  DOUBLE,
            rank_score DOUBLE
        )
    """)
    
    # 使用 DataFrame 批量插入提高性能
    con.execute("INSERT INTO tft_alpha_scores SELECT * FROM df_results")
    
    # 验证数据
    final_count = con.execute("SELECT COUNT(*) FROM tft_alpha_scores").fetchone()[0]
    sample_data = con.execute("SELECT * FROM tft_alpha_scores WHERE date = (SELECT MAX(date) FROM tft_alpha_scores) LIMIT 5").fetchdf()
    con.close()

    print("-" * 60)
    print(f"SUCCESS!")
    print(f"Total Records: {final_count:,}")
    print(f"Inference Time: {_time.time()-t0:.1f}s")
    print("\nSample of latest scores:")
    print(sample_data)
    print("-" * 60)

if __name__ == "__main__":
    main()