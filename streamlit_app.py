import streamlit as st
import pandas as pd
import json
import plotly.express as px
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Functional Depth Ranking", layout="wide")
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #555; margin-bottom: 2rem; }
    .universe-title { font-size: 1.5rem; font-weight: 600; margin-top: 1rem; margin-bottom: 1rem; padding-left: 0.5rem; border-left: 5px solid #1f77b4; }
    .etf-card { background: linear-gradient(135deg, #1f77b4 0%, #2c3e50 100%); color: white; border-radius: 15px; padding: 1rem; margin: 0.5rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.2); transition: transform 0.2s; }
    .etf-card:hover { transform: translateY(-5px); }
    .etf-ticker { font-size: 1.3rem; font-weight: bold; }
    .etf-score { font-size: 1rem; margin-top: 0.3rem; }
    .positive { color: #00cc96; }
    .negative { color: #ef553b; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 Functional Depth Ranking Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Modified Band Depth (MBD) on rolling 63‑day return curves | Score = depth × annualised return</div>', unsafe_allow_html=True)

st.sidebar.markdown("## 📊 Functional Depth")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Run Date:** `{st.session_state.get('run_date', 'Not loaded')}`")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown("**Method:** Modified Band Depth (López‑Pintado & Romo, 2009)")
st.sidebar.markdown(f"**Curve window:** {config.ROLLING_WINDOW} days")
st.sidebar.markdown("---")
st.sidebar.caption("Deepest curve = functional median (most central). Score = depth * annualised return – ranks stable & high‑return ETFs.")

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return files
    except Exception as e:
        return [f"Error: {e}"]

def find_latest_json(files):
    json_files = [f for f in files if f.endswith('.json') and 'functional_depth_' in f]
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

files = list_repo_files()
latest = find_latest_json(files)
if not latest:
    st.error("No results found. Run trainer first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error: {data['error']}")
    st.stop()

st.session_state['run_date'] = data['run_date']
universes = data["universes"]

st.header("🏆 Top ETFs by Functional Depth × Return")
st.markdown("*Higher score = deeper (more central) + higher annualised return.*")

for universe_name, uni_data in universes.items():
    top_etfs = uni_data.get("top_etfs", [])
    if not top_etfs:
        continue
    st.markdown(f'<div class="universe-title">{universe_name.replace("_", " ").title()}</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, etf in enumerate(top_etfs):
        with cols[idx]:
            score = etf["score"]
            depth = etf["depth"]
            ann_ret = etf["annual_return"]
            st.markdown(f"""
            <div class="etf-card">
                <div class="etf-ticker">{etf['ticker']}</div>
                <div class="etf-score">score = {score:.3f}</div>
                <div class="etf-score">depth = {depth:.3f} | ann ret = {ann_ret:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
    # Optional expander with full ranking table
    with st.expander(f"📋 Full ranking for {universe_name}"):
        # Recompute from data? We don't have full depth values in the JSON. We could store all.
        # Instead, we can just display the top 3; but we can also fetch all from the JSON if we stored earlier.
        # For simplicity, we show only the top 3. To add full ranking, we would need to store full depth array.
        st.write("Top 3 shown above. Full depth table not stored to keep JSON lightweight.")
    st.divider()

st.caption("Modified Band Depth (MBD) measures how central the ETF's 63‑day return curve is among all ETFs. Deeper = more stable/central. Score combines depth with annualised return.")
