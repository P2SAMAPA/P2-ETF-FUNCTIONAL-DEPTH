"""
Daily training: compute functional depth on last 63‑day return curves,
score = depth * annualized return, output top 3 ETFs per universe.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from depth_ranking import compute_scores

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < config.ROLLING_WINDOW + 1:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Compute scores on the most recent window
        scores_df = compute_scores(returns, window=config.ROLLING_WINDOW, depth_method=config.DEPTH_METHOD)
        top3 = scores_df.head(config.TOP_N)
        top_etfs = [
            {"ticker": row['ETF'], "score": float(row['score']), 
             "depth": float(row['depth']), "annual_return": float(row['annual_return'])}
            for _, row in top3.iterrows()
        ]
        print(f"  Top 3 ETFs (depth, ann_return, score):")
        for etf in top_etfs:
            print(f"    {etf['ticker']}: depth={etf['depth']:.3f}, ann_ret={etf['annual_return']:.3f}, score={etf['score']:.3f}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "run_date": today
        }

    # Save results
    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/functional_depth_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Functional Depth Ranking complete ===")

if __name__ == "__main__":
    main()
