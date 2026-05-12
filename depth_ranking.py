"""
Functional Depth Ranking using Modified Band Depth (MBD).
For each ETF curve (rolling window), compute depth = fraction of curve pairs whose band contains it.
Score = depth * annualized return.
"""
import numpy as np
import pandas as pd

def modified_band_depth(curves):
    """
    curves: numpy array of shape (n_assets, window_length) – each row is a return curve.
    Returns: depth for each curve (array length n_assets).
    MBD for curve i: proportion of pairs (j,k) such that min(curves[j], curves[k]) <= curves[i] <= max(curves[j], curves[k]) pointwise.
    """
    n, L = curves.shape
    if n < 2:
        return np.ones(n)
    depth = np.zeros(n)
    # Precompute all pairs (j,k) with j<k
    pairs = [(j, k) for j in range(n) for k in range(j+1, n)]
    total_pairs = len(pairs)
    if total_pairs == 0:
        return np.ones(n)
    for i in range(n):
        count = 0
        curve_i = curves[i]
        for j, k in pairs:
            lower = np.minimum(curves[j], curves[k])
            upper = np.maximum(curves[j], curves[k])
            if np.all((curve_i >= lower) & (curve_i <= upper)):
                count += 1
        depth[i] = count / total_pairs
    return depth

def band_depth(curves):
    """
    Original band depth (simpler): fraction of pairs where the curve is inside the band.
    Same as modified but uses only one ordering? For consistency, we use modified.
    """
    return modified_band_depth(curves)

def compute_scores(returns_df, window=63, depth_method='modified_band'):
    """
    returns_df: DataFrame with dates as index, ETFs as columns.
    For the most recent window (last `window` days), compute depth for each ETF,
    then compute annualized return = mean daily return * 252.
    Score = depth * annualized_return.
    Returns a DataFrame with columns: ETF, depth, annual_return, score.
    """
    if len(returns_df) < window:
        raise ValueError(f"Insufficient data: need at least {window} days")
    # Take last `window` rows
    window_data = returns_df.iloc[-window:].values.T   # shape (n_assets, window)
    # Compute depth
    if depth_method == 'modified_band':
        depth = modified_band_depth(window_data)
    else:
        depth = band_depth(window_data)
    # Compute annualized return (mean daily return * 252)
    mean_daily = returns_df.iloc[-window:].mean(axis=0).values
    annual_return = mean_daily * 252   # simple, no compounding
    # Score
    score = depth * annual_return
    # Build result DataFrame
    result = pd.DataFrame({
        'ETF': returns_df.columns,
        'depth': depth,
        'annual_return': annual_return,
        'score': score
    })
    return result.sort_values('score', ascending=False)
