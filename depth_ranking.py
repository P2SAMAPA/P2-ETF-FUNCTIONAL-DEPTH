"""
Functional Depth Ranking using Modified Band Depth (MBD) for multiple windows.
For each window, compute depth and annualised return. For each ETF, pick the window that maximizes score.
"""
import numpy as np
import pandas as pd

def modified_band_depth(curves):
    """
    curves: numpy array of shape (n_assets, window_length)
    Returns: depth for each curve (array length n_assets).
    """
    n, L = curves.shape
    if n < 2:
        return np.ones(n)
    depth = np.zeros(n)
    pairs = [(j, k) for j in range(n) for k in range(j+1, n)]
    total_pairs = len(pairs)
    if total_pairs == 0:
        return np.ones(n)
    for i in range(n):
        curve_i = curves[i]
        count = 0
        for j, k in pairs:
            lower = np.minimum(curves[j], curves[k])
            upper = np.maximum(curves[j], curves[k])
            if np.all((curve_i >= lower) & (curve_i <= upper)):
                count += 1
        depth[i] = count / total_pairs
    return depth

def compute_scores_for_window(returns_df, window, depth_method='modified_band'):
    """
    Compute depth and annualised return for a single window.
    Returns DataFrame with columns: ETF, depth, annual_return, score.
    """
    if len(returns_df) < window:
        return pd.DataFrame(columns=['ETF', 'depth', 'annual_return', 'score'])
    window_data = returns_df.iloc[-window:].values.T   # (n_assets, window)
    depth = modified_band_depth(window_data)
    mean_daily = returns_df.iloc[-window:].mean(axis=0).values
    annual_return = mean_daily * 252
    score = depth * annual_return
    return pd.DataFrame({
        'ETF': returns_df.columns,
        'depth': depth,
        'annual_return': annual_return,
        'score': score
    })

def compute_best_scores(returns_df, windows=[63, 126, 252], depth_method='modified_band'):
    """
    For each ETF, compute the maximum score across all windows.
    Returns a DataFrame with columns: ETF, best_score, best_window, depth, annual_return (for that window).
    """
    all_windows_scores = []
    for win in windows:
        df_win = compute_scores_for_window(returns_df, win, depth_method)
        if df_win.empty:
            continue
        df_win['window'] = win
        all_windows_scores.append(df_win)
    if not all_windows_scores:
        return pd.DataFrame(columns=['ETF', 'best_score', 'best_window', 'depth', 'annual_return'])
    combined = pd.concat(all_windows_scores, ignore_index=True)
    # For each ETF, find row with max score
    best_idx = combined.groupby('ETF')['score'].idxmax()
    best = combined.loc[best_idx].copy()
    best = best.rename(columns={'score': 'best_score', 'window': 'best_window'})
    return best[['ETF', 'best_score', 'best_window', 'depth', 'annual_return']]
