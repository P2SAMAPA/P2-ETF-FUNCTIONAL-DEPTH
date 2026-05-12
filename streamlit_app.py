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
    .etf-score { font-size: 0.9rem; margin-top: 0.2rem; }
    .positive { color: #00cc96; }
    .negative { color: #ef553b; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 Functional Depth Ranking Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Modified Band Depth (MBD) on 63/126/252‑day return curves | Score = depth × annualised return | Best window selected per ETF</div>', unsafe_allow_html=True)

st.sidebar.markdown("## 📊 Functional Depth")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Run Date:** `{st.session_state.get('run_date', 'Not loaded')}`")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown("**Method:** Modified Band Depth (López‑Pintado & Romo, 2009)")
st.sidebar.markdown("**Windows:** 63, 126, 252 days (best per ETF)")
st.sidebar.markdown("---")
st.sidebar.caption("Deepest curve = functional median (most central). Score = depth × annualised return – ranks stable & high‑return ETFs.")

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

st.header("🏆 Top ETFs by Functional Depth × Return (Best Window)")
st.markdown("*Each ETF's best window (63, 126, or 252 days) is selected to maximise score = depth × annualised return.*")

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
            window = etf["window"]
            st.markdown(f"""
            <div class="etf-card">
                <div class="etf-ticker">{etf['ticker']}</div>
                <div class="etf-score">score = {score:.3f}</div>
                <div class="etf-score">depth = {depth:.3f} | window = {window}d</div>
                <div class="etf-score">ann ret = {ann_ret:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
    # Optional expander – we can show the chosen window distribution
    with st.expander(f"📋 Details for {universe_name}"):
        st.write("**Top 3 ETFs – Best window per ETF**")
        st.dataframe(pd.DataFrame(top_etfs), use_container_width=True)
    st.divider()

st.caption("Modified Band Depth (MBD) measures how central the ETF's return curve is among all ETFs. For each ETF, we compute depth and annualised return over 63, 126, and 252 days, then keep the window with highest product. The final ranking uses that best score.")
