"""
Daily training: compute scores for 63, 126, 252 days windows, pick best per ETF, rank.
Output top 3 ETFs per universe with chosen window.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from depth_ranking import compute_best_scores

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")
    windows = [63, 126, 252]   # defined here or in config

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < max(windows) + 1:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Compute best scores across windows
        best_df = compute_best_scores(returns, windows=windows, depth_method=config.DEPTH_METHOD)
        if best_df.empty:
            print("  No valid scores")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Sort by best_score descending, take top 3
        top3 = best_df.sort_values('best_score', ascending=False).head(config.TOP_N)
        top_etfs = []
        for _, row in top3.iterrows():
            top_etfs.append({
                "ticker": row['ETF'],
                "score": float(row['best_score']),
                "depth": float(row['depth']),
                "annual_return": float(row['annual_return']),
                "window": int(row['best_window'])
            })
        print(f"  Top 3 ETFs (window, depth, ann_return, best_score):")
        for etf in top_etfs:
            print(f"    {etf['ticker']}: win={etf['window']}, depth={etf['depth']:.3f}, ann_ret={etf['annual_return']:.3f}, score={etf['score']:.3f}")
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
    print("\n=== Functional Depth Ranking (multi‑window) complete ===")

if __name__ == "__main__":
    main()
