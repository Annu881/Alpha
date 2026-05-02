"""
BTC Forecast Dashboard  ·  Binance Live Clone
==============================================
"""

import json
import time
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data_fetch import fetch_klines, closes_array, timestamps_array
from model import fit_and_predict
from backtest import evaluate

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Binance Oracle",
    page_icon="🔶",
    layout="wide",
    initial_sidebar_state="collapsed", 
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Teko:wght@500;600&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #0b0e11; 
    color: #b7bdc6; 
}

.main { background: #0b0e11 !important; }

/* Edge to edge */
.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 0rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    max-width: 100% !important;
}

/* Top Ticker */
.ticker-bar {
    display: flex;
    align-items: center;
    background-color: #181a20;
    border-bottom: 1px solid #2b3139;
    padding: 0.5rem 1rem;
    margin-bottom: 0.5rem;
    gap: 2rem;
}
.t-brand {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    color: #fcd535; 
    font-size: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.t-pair {
    font-size: 1.25rem;
    font-weight: 600;
    color: #eaecef;
}
.t-price {
    font-size: 1.25rem;
    font-weight: 600;
    color: #0ecb81; 
}
.t-price.down {
    color: #f6465d; 
}
.t-stat {
    display: flex;
    flex-direction: column;
}
.t-lbl {
    font-size: 0.65rem;
    color: #848e9c;
}
.t-val {
    font-size: 0.8rem;
    color: #eaecef;
    font-weight: 500;
}

/* Right Panel (Orderbook style) */
.right-panel {
    background-color: #181a20;
    border: 1px solid #2b3139;
    padding: 1rem;
    height: 100%;
}
.rp-header {
    font-size: 0.85rem;
    color: #eaecef;
    font-weight: 600;
    margin-bottom: 1rem;
    border-bottom: 1px solid #2b3139;
    padding-bottom: 0.5rem;
}

.orderbook-row {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    margin-bottom: 0.2rem;
}
.ob-red { color: #f6465d; }
.ob-green { color: #0ecb81; }
.ob-gray { color: #848e9c; }

/* Custom Sliders for Binance feel */
.stSlider label { color: #848e9c !important; font-size: 0.7rem !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #2b3139;
    padding: 0;
    gap: 1.5rem;
}
.stTabs [data-baseweb="tab"] {
    color: #848e9c !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #eaecef !important;
    border-bottom: 2px solid #fcd535 !important;
}

header[data-testid="stHeader"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Simulator Session State ─────────────────────────────────────────────────
if "positions" not in st.session_state: st.session_state.positions = []
if "open_orders" not in st.session_state: st.session_state.open_orders = []
if "trade_history" not in st.session_state: st.session_state.trade_history = []
if "balance" not in st.session_state: st.session_state.balance = 5000.00
if "o_size" not in st.session_state: st.session_state["o_size"] = 0.100


# ── Persistence ─────────────────────────────────────────────────────────────
HISTORY_FILE = Path("prediction_history.jsonl")

def load_history() -> list:
    if not HISTORY_FILE.exists(): return []
    records = []
    with HISTORY_FILE.open() as f:
        for line in f:
            if line.strip():
                try: records.append(json.loads(line))
                except: pass
    return records

def save_prediction(record: dict):
    with HISTORY_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")

def fill_actuals(history: list, current_candles: list) -> list:
    ts_to_close = {c["open_time"]: c["close"] for c in current_candles}
    updated = []
    for rec in history:
        if rec.get("actual_close") is None:
            target_ts = rec.get("predicted_bar_open_time")
            if target_ts and target_ts in ts_to_close:
                rec = dict(rec)
                rec["actual_close"] = ts_to_close[target_ts]
                rec["hit"] = int(rec["lower_95"] <= rec["actual_close"] <= rec["upper_95"])
        updated.append(rec)
    return updated

# ── Fetch Data ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=1)  # Ultra Live Real-Time Tick (1s)
def get_live_data():
    candles = fetch_klines(limit=520)
    closes = closes_array(candles)
    timestamps = timestamps_array(candles)
    return candles, closes, timestamps

@st.cache_data(ttl=3600)
def load_backtest_metrics():
    bt_file = Path("backtest_results.jsonl")
    if not bt_file.exists(): return None
    preds = []
    with bt_file.open() as f:
        for line in f:
            if line.strip():
                try: preds.append(json.loads(line.strip()))
                except: pass
    if not preds: return None
    return evaluate(preds)

try:
    candles, closes, timestamps = get_live_data()
    current_price = float(closes[-1])
    prev_price = float(closes[-2])
    price_delta = current_price - prev_price
    pct_delta = (price_delta / prev_price) * 100
    current_ts = int(timestamps[-1])
    next_bar_ts = current_ts + 60 * 1000  # 1 MINUTE PREDICTIONS NOW FOR LIVE ACTION
    
    # 24H stats (roughly 1440 mins = 24H, but limit is 520, so using total range available)
    day_closes = float(closes[-500]) if len(closes) > 500 else float(closes[0])
    h24_chg = current_price - day_closes
    h24_pct = (h24_chg / day_closes) * 100
    h24_h = max([float(c["high"]) for c in candles])
    h24_l = min([float(c["low"]) for c in candles])
    h24_v = sum([float(c["volume"]) for c in candles])
except Exception as e:
    st.error(f"Data feed lost: {e}")
    st.stop()


# ── Paper Trading Execution Engine Run Tick  ────────────────────────────────
filled_this_tick = []
remaining_orders = []
for order in st.session_state.open_orders:
    p = float(order["Price"])
    executed = False
    
    if order["Type"] == "Limit":
        if order["Side"] == "Buy" and current_price <= p:
            executed = True
        elif order["Side"] == "Sell" and current_price >= p:
            executed = True
    elif order["Type"] == "Stop Limit":
        stop = float(order["StopPrice"])
        if order["Side"] == "Buy" and current_price >= stop:
            executed = True
        elif order["Side"] == "Sell" and current_price <= stop:
            executed = True
            
    if executed:
        order["Status"] = "Filled"
        order["Executed Time"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        filled_this_tick.append(order)
        
        # Build Position logic
        pos_idx = next((i for i, pos in enumerate(st.session_state.positions) if pos["Symbol"] == "BTCUSDT Perp"), -1)
        if pos_idx >= 0:
            pos = st.session_state.positions[pos_idx]
            if pos["Side"] == order["Side"]:
                # Add to position
                tot_sz = pos["Size"] + order["Amount"]
                avg_en = ((pos["Entry"] * pos["Size"]) + (p * order["Amount"])) / tot_sz
                st.session_state.positions[pos_idx]["Size"] = tot_sz
                st.session_state.positions[pos_idx]["Entry"] = avg_en
            else:
                # Reduce/Close position
                if order["Amount"] >= pos["Size"]:
                    # Closed entirely
                    st.session_state.positions.pop(pos_idx)
                else:
                    st.session_state.positions[pos_idx]["Size"] -= order["Amount"]
        else:
            st.session_state.positions.append({
                "Id": order["Id"],
                "Symbol": "BTCUSDT Perp",
                "Side": order["Side"],
                "Size": order["Amount"],
                "Entry": p
            })
    else:
        remaining_orders.append(order)

st.session_state.open_orders = remaining_orders
if filled_this_tick:
    for filled in filled_this_tick:
        st.session_state.trade_history.append(filled)
        st.toast(f"✅ {filled['Side']} {filled['Type']} Triggered @ ${filled['Price']:,.1f}!", icon="⚡")


# ── Ticker Header ───────────────────────────────────────────────────────────
p_col = "t-price" if price_delta >= 0 else "t-price down"
s_col = "#0ecb81" if h24_chg >= 0 else "#f6465d"
sign = "+" if h24_chg >= 0 else ""

st.markdown(f"""
<div class='ticker-bar'>
    <div class='t-brand'>🔶 ORACLE</div>
    <div style='display:flex; align-items:baseline; gap:1rem;'>
        <div class='t-pair'>BTCUSDT Perpetual</div>
        <div class='{p_col}'>${current_price:,.1f}</div>
    </div>
    <div class='t-stat'>
        <div class='t-lbl'>24h Change</div>
        <div class='t-val' style='color:{s_col};'>{sign}{h24_chg:,.1f} {sign}{h24_pct:.2f}%</div>
    </div>
    <div class='t-stat'>
        <div class='t-lbl'>24h High</div>
        <div class='t-val'>${h24_h:,.1f}</div>
    </div>
    <div class='t-stat'>
        <div class='t-lbl'>24h Low</div>
        <div class='t-val'>${h24_l:,.1f}</div>
    </div>
    <div class='t-stat'>
        <div class='t-lbl'>24h Vol(BTC)</div>
        <div class='t-val'>{h24_v:,.2f}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Layout ──────────────────────────────────────────────────────────────────
col_chart, col_side = st.columns([0.75, 0.25], gap="small")

with col_side:
    st.markdown("<div class='right-panel'>", unsafe_allow_html=True)
    
    # 1. Oracle Order Book (Compact)
    st.markdown("<div class='rp-header'>Oracle Order Book</div>", unsafe_allow_html=True)
    vol_win = st.slider("Vol Lookback", 6, 72, 24, label_visibility="collapsed")
    df_t = st.slider("Tail Thickness", 3, 30, 4, label_visibility="collapsed")
    lower, upper, sigma_bar = fit_and_predict(closes, n_sims=5000, vol_window=vol_win, df_t=float(df_t), confidence=0.95)
    
    st.markdown(f"<div class='orderbook-row' style='background:rgba(246,70,93,0.1); margin-top:0.5rem;'><span class='ob-red' style='font-weight:700;'>{upper:,.1f}</span><span class='ob-gray'>Oracle Target</span><span class='ob-red'>UPPER</span></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='margin: 0.5rem 0; font-size:1.5rem; font-family:"Teko", sans-serif; line-height:1; color:{s_col}; display:flex; align-items:center; gap:0.5rem;'>
        {current_price:,.1f} <span style='font-size:0.8rem; font-family:"Inter", sans-serif; color:#848e9c;'>Spread: ${(upper-lower):,.1f}</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"<div class='orderbook-row' style='background:rgba(14,203,129,0.1); margin-bottom:1rem;'><span class='ob-green' style='font-weight:700;'>{lower:,.1f}</span><span class='ob-gray'>Oracle Target</span><span class='ob-green'>LOWER</span></div>", unsafe_allow_html=True)
    
    # 2. Place Order Panel (Mimicking Binance)
    st.markdown("<div class='rp-header' style='margin-top:2rem;'>Place Order</div>", unsafe_allow_html=True)
    order_type = st.segmented_control("type", ["Limit", "Market", "Stop Limit"], default="Limit", label_visibility="collapsed")
    st.markdown(f"<div style='font-size:0.7rem; color:#848e9c; margin-bottom:0.1rem;'>Avail <span style='color:#eaecef;'>{st.session_state.balance:,.2f} USDT</span></div>", unsafe_allow_html=True)
    
    o_price = 0.0
    stop_price = 0.0
    
    if order_type in ["Limit", "Stop Limit"]:
        if "o_price" not in st.session_state: st.session_state["o_price"] = float(current_price)
        if order_type == "Stop Limit":
            stop_price = st.number_input("Stop (USDT)", value=float(current_price), format="%.1f", step=0.5)
        o_price = st.number_input("Price (USDT)", key="o_price", format="%.1f", step=0.5)
    else:
        st.text_input("Price (USDT)", value="Market", disabled=True)
        o_price = float(current_price)
        
    o_size = st.number_input("Size (BTC)", key="o_size", format="%.3f", step=0.010)
    st.slider("Size %", 0, 100, 0, label_visibility="collapsed")
    
    st.markdown("""
    <style>
    div.stButton > button[kind="primary"] { background-color: #0ecb81 !important; color: white !important; font-size:1.1rem !important; border-radius:4px !important; }
    div.stButton > button[kind="secondary"] { background-color: #f6465d !important; color: white !important; font-size:1.1rem !important; border-radius:4px !important; }
    </style>
    """, unsafe_allow_html=True)
    
    b1, b2 = st.columns(2)
    
    def process_button_press(side):
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "Id": str(uuid.uuid4())[:8], "Time": now_str, 
            "Symbol": "BTCUSDT Perp", "Type": order_type, "Side": side,
            "Price": round(o_price,2), "Amount": o_size, 
            "Status": "Open", "StopPrice": round(stop_price,2) if order_type=="Stop Limit" else "-"
        }
        if order_type == "Market":
            # Execute instantly
            record["Status"] = "Filled"
            record["Executed Time"] = now_str
            st.session_state.trade_history.append(record)
            
            # Update Position
            pos_idx = next((i for i, pos in enumerate(st.session_state.positions) if pos["Symbol"] == "BTCUSDT Perp"), -1)
            if pos_idx >= 0:
                pos = st.session_state.positions[pos_idx]
                if pos["Side"] == side:
                    tot_sz = pos["Size"] + o_size
                    avg_en = ((pos["Entry"] * pos["Size"]) + (o_price * o_size)) / tot_sz
                    st.session_state.positions[pos_idx]["Size"] = tot_sz
                    st.session_state.positions[pos_idx]["Entry"] = avg_en
                else: # Opposite Side
                    if o_size >= pos["Size"]:
                        st.session_state.balance += ((current_price - pos["Entry"]) if pos["Side"]=="Buy" else (pos["Entry"] - current_price)) * pos["Size"]
                        st.session_state.positions.pop(pos_idx)
                    else:
                        st.session_state.positions[pos_idx]["Size"] -= o_size
            else:
                st.session_state.positions.append({"Id": record["Id"], "Symbol": "BTCUSDT Perp", "Side": side, "Size": o_size, "Entry": o_price})
            
            st.toast(f"✅ Market {side} Filled for {o_size} BTC!", icon="⚡")
        else:
            st.session_state.open_orders.append(record)
            st.toast(f"✅ {order_type} {side} placed for {o_size} BTC @ ${o_price:,.1f}", icon="📩")

    if b1.button("Buy/Long", type="primary", use_container_width=True): process_button_press("Buy")
    if b2.button("Sell/Short", type="secondary", use_container_width=True): process_button_press("Sell")
    
    st.markdown("<hr style='border-color:#2b3139; margin:1.5rem 0 1rem 0;'>", unsafe_allow_html=True)
    is_live = st.toggle("🔴 LIVE AUTO-REFRESH", value=True, help="Auto-updates via loop every 1 second!")
    
    st.markdown("</div>", unsafe_allow_html=True)


with col_chart:
    hist_candles = 90
    chart_candles = candles[-hist_candles:]
    df = pd.DataFrame(chart_candles)
    df["dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

    n = len(closes)
    ribbon_lower, ribbon_upper = [], []
    for i in range(n - hist_candles, n):
        hist_slice = closes[: i + 1]
        if len(hist_slice) >= vol_win + 1:
            lo, hi, _ = fit_and_predict(hist_slice, vol_window=vol_win, n_sims=500, df_t=float(df_t))
        else:
            lo = hi = float(closes[i])
        ribbon_lower.append(lo)
        ribbon_upper.append(hi)

    df["ribbon_lower"] = ribbon_lower
    df["ribbon_upper"] = ribbon_upper
    next_dt = pd.Timestamp(next_bar_ts, unit="ms", tz="UTC")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_width=[0.2, 0.7])

    fig.add_trace(go.Scatter(
        x=list(df["dt"]) + list(df["dt"])[::-1], y=list(df["ribbon_upper"]) + list(df["ribbon_lower"])[::-1],
        fill="toself", fillcolor="rgba(252,213,53,0.05)", line=dict(color="rgba(0,0,0,0)"), name="95% CI ribbon", hoverinfo="skip",
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df["dt"], y=df["ribbon_upper"], line=dict(color="rgba(252,213,53,0.4)", width=1, dash="dot"), name="CI upper"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["dt"], y=df["ribbon_lower"], line=dict(color="rgba(252,213,53,0.4)", width=1, dash="dot"), name="CI lower"), row=1, col=1)

    fig.add_trace(go.Candlestick(
        x=df["dt"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="BTCUSDT", increasing_line_color="#0ecb81", decreasing_line_color="#f6465d",
    ), row=1, col=1)

    fig.add_shape(type="rect",
        x0=next_dt, x1=next_dt + pd.Timedelta(minutes=1),
        y0=lower, y1=upper,
        fillcolor="rgba(252,213,53,0.1)", line=dict(color="#fcd535", width=1.5),
        row=1, col=1
    )

    colors = ['#0ecb81' if df.iloc[i]['close'] >= df.iloc[i]['open'] else '#f6465d' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df['dt'], y=df['volume'], marker_color=colors, showlegend=False), row=2, col=1)

    fig.update_layout(
        paper_bgcolor="#181a20", plot_bgcolor="#181a20",
        font=dict(family="JetBrains Mono", color="#848e9c"),
        xaxis=dict(gridcolor="#2b3139", showgrid=True, rangeslider=dict(visible=False)),
        yaxis=dict(gridcolor="#2b3139", showgrid=True, tickformat="$,.0f", side="right"),
        xaxis2=dict(gridcolor="#2b3139", showgrid=False, rangeslider=dict(visible=False)),
        yaxis2=dict(gridcolor="#2b3139", showgrid=False, tickformat=".2s", side="right"),
        margin=dict(l=0, r=0, t=10, b=0),
        height=650, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"), hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Bottom Panel (Positions & History) ──────────────────────────────────
    pred_record = {
        "fetched_at": int(time.time() * 1000), "current_bar_open_time": current_ts, "predicted_bar_open_time": next_bar_ts,
        "current_price": round(current_price, 2), "lower_95": round(lower, 2), "upper_95": round(upper, 2),
        "sigma_used": round(sigma_bar, 8), "actual_close": None, "hit": None,
    }
    history = load_history()
    existing_ts = {r["current_bar_open_time"] for r in history}
    if pred_record["current_bar_open_time"] not in existing_ts:
        save_prediction(pred_record)
        history.append(pred_record)
    
    history = fill_actuals(history, candles)
    if any(r.get("actual_close") is not None for r in history):
        with HISTORY_FILE.open("w") as f:
            for r in history:
                f.write(json.dumps(r) + "\n")
    
    t_pos, t_open, t_ord, t_trade, t_ora, t_val = st.tabs([
        f"Positions({len(st.session_state.positions)})", f"Open Orders({len(st.session_state.open_orders)})", 
        "Order History", "Trade History", "Oracle Ledger", "Backtest Metrics"
    ])
    
    with t_pos:
        if not st.session_state.positions:
            st.markdown("<div style='font-size:0.85rem; color:#848e9c; padding:1rem;'>No open positions.</div>", unsafe_allow_html=True)
        else:
            for p in st.session_state.positions:
                side_color = "#0ecb81" if p["Side"] == "Buy" else "#f6465d"
                pnl = (current_price - p["Entry"]) * p["Size"] if p["Side"] == "Buy" else (p["Entry"] - current_price) * p["Size"]
                margin_req = p["Entry"] * p["Size"] * 0.05
                roe = (pnl / margin_req) * 100 if margin_req > 0 else 0
                pnl_color = "#0ecb81" if pnl >= 0 else "#f6465d"
                sign = "+" if pnl >= 0 else ""
                
                st.markdown(f"""
                <div style='font-family:"Inter",sans-serif; font-size:0.85rem; padding:0.5rem 0; margin-bottom:0.5rem; border-bottom:1px dashed #2b3139; color:#848e9c; display:flex; gap:3rem; overflow-x:auto;'>
                    <div style='min-width:150px; color:#eaecef;'>Symbol<br><span style='color:{side_color}; font-weight:700;'>BTCUSDT Perp</span> <span style='font-size:11px; background:#2b3139; padding:2px 4px; border-radius:2px;'>20x</span></div>
                    <div style='min-width:100px;'>Size<br><span style='color:{side_color};'>{p['Size']:.4f}</span></div>
                    <div style='min-width:100px;'>Entry Price<br>${p['Entry']:,.1f}</div>
                    <div style='min-width:100px;'>Mark Price<br>${current_price:,.1f}</div>
                    <div style='min-width:100px;'>Liq. Price<br>${max(0, p['Entry'] - 3800) if p['Side']=="Buy" else p['Entry']+3800:,.1f}</div>
                    <div style='min-width:100px;'>Margin Ratio<br>8.44%</div>
                    <div style='min-width:150px;'>PNL (ROE%)<br><span style='color:{pnl_color};'>{sign}${pnl:,.2f} ({sign}{roe:.1f}%)</span></div>
                    <div style='min-width:80px;'>Action<br><span style='color:#fcd535; cursor:pointer;'>[Close Position]</span></div>
                </div>
                """, unsafe_allow_html=True)

    def draw_df(records, fields):
        if not records:
            st.markdown("<div style='font-size:0.85rem; color:#848e9c; padding:1rem;'>No records found.</div>", unsafe_allow_html=True)
            return
        df = pd.DataFrame(sorted(records, key=lambda x: x["Time"], reverse=True))[fields]
        def styles(v):
            if v == 'Buy': return 'color: #0ecb81'
            if v == 'Sell': return 'color: #f6465d'
            if v == 'Open': return 'color: #fcd535'
            if v == 'Filled': return 'color: #0ecb81'
            return ''
        cols = [c for c in ['Side', 'Status'] if c in df.columns]
        st.dataframe(df.style.map(styles, subset=cols) if cols else df, use_container_width=True, hide_index=True)

    with t_open:
        draw_df(st.session_state.open_orders, ["Time", "Symbol", "Type", "Side", "Price", "StopPrice", "Amount", "Status"])

    with t_ord:
        # All orders including open and filled
        all_orders = st.session_state.open_orders + st.session_state.trade_history
        draw_df(all_orders, ["Time", "Symbol", "Type", "Side", "Price", "Amount", "Status"])

    with t_trade:
        draw_df(st.session_state.trade_history, ["Executed Time", "Symbol", "Side", "Price", "Amount", "Status"])

    with t_ora:
        settled = [r for r in history if r.get("actual_close") is not None]
        if settled:
            df_hist = pd.DataFrame([{
                "Resolved Time": datetime.fromtimestamp(r["current_bar_open_time"]/1000, tz=timezone.utc).strftime("%m-%d %H:%M"),
                "Spread Captured": f"${r['upper_95'] - r['lower_95']:,.0f}",
                "Predicted Bottom": f"${r['lower_95']:,.1f}",
                "Actual Final Close": f"${r['actual_close']:,.1f}",
                "Predicted Top": f"${r['upper_95']:,.1f}",
                "Trade Outcome": "✅ SUCCESS" if r.get("hit") else "❌ EXCEEDED",
            } for r in sorted(settled, key=lambda x: x["current_bar_open_time"], reverse=True)])
            st.dataframe(df_hist, use_container_width=True, hide_index=True)
        else:
            st.info("Awaiting next 1m close to resolve oracle logic bounds.")

    with t_val:
        bt_metrics = load_backtest_metrics()
        if bt_metrics:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Oracle Coverage", f"{bt_metrics['coverage_95']:.3f} / 0.95")
            c2.metric("Miss Breach Rate", f"{(1 - bt_metrics['coverage_95'])*100:.1f}%")
            c3.metric("Avg Predicted Vol Range", f"${bt_metrics['avg_width']:,.0f}")
            c4.metric("Winkler Penalization Score", f"{bt_metrics['mean_winkler_95']:,.0f}")

if is_live:
    time.sleep(1)
    st.rerun()

