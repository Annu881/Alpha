# 🔶 Binance Oracle — Google Colab Instructions

## Step 1: Upload the Notebook
1. Go to **https://colab.research.google.com**
2. Click **File → Upload Notebook**
3. Upload `Binance_Oracle_Submission.ipynb` from your Alpha folder

---

## Step 2: Run Cells in Order (Top → Bottom)

### Cell 1 — Install Dependencies
```python
!pip install streamlit plotly pandas numpy scipy
```
✅ Wait until you see `Successfully installed`

---

### Cell 2 — Write `data_fetch.py`
```python
%%writefile data_fetch.py
...
```
✅ You'll see: `Writing data_fetch.py`

---

### Cell 3 — Write `model.py`
```python
%%writefile model.py
...
```
✅ You'll see: `Writing model.py`

---

### Cell 4 — Write `backtest.py`
```python
%%writefile backtest.py
...
```
✅ You'll see: `Writing backtest.py`

---

### Cell 5 — Write `app.py`
```python
%%writefile app.py
...
```
✅ You'll see: `Writing app.py`

---

### Cell 6 — Run Backtest (Generates Oracle Metrics)
```python
!python backtest.py
```
✅ Output will look like:
```
Coverage (target 0.95):  0.9412
Average width ($):       823.45
Mean Winkler score:      1204.32
Results saved → backtest_results.jsonl
```
⚠️ This may take **2–3 minutes**, please wait!

---

### Cell 7 — Launch the Binance Dashboard (LAST CELL)
```python
from google.colab import output
output.serve_kernel_port_as_window(8501)
!streamlit run app.py --server.port 8501
```
✅ Colab will **automatically open a new browser tab** with your live dashboard!

> ⚠️ If nothing opens: look at the top of your browser for a **"Pop-up blocked"** icon → click it → Allow

---

## 🎯 What You Will See
- **Live BTCUSDT candlestick chart** updating every second
- **Oracle Confidence Band** (yellow ribbon) showing next 1-minute prediction
- **Right panel:** Order Book + Place Order (Limit / Market / Stop Limit)
- **Bottom tabs:** Positions | Open Orders | Order History | Trade History | Oracle Ledger | Backtest Metrics

---

## 🚨 Important Notes
- Cells must be run **in order** (1 → 7)
- Do NOT skip Cell 6 (backtest) — it generates the metrics shown in the dashboard
- The last cell (Cell 7) will run **forever** (that is normal — the dashboard needs it running)
- To stop: click the **Stop (■)** button next to the cell
