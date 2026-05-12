# Functional Depth Ranking Engine

**Modified Band Depth (MBD)** on rolling 63‑day return curves.  
Measures how central each ETF's return trajectory is within the universe.  
Score = depth × annualised return → ranks stable, high‑return ETFs.

- Rolling window: 63 days
- Output: top 3 ETFs per universe (FI, Equity, Combined)
- Daily retraining (GitHub Actions)
- Results stored on Hugging Face Hub

## Run locally
```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
