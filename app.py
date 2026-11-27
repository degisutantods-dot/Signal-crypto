import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

API = "https://api.gateio.ws/api/v4"
HEADERS = {"Accept": "application/json", "User-Agent": "GateMarketIntelligence/3.2"}
TFS = ["15m", "1h", "4h"]
WEIGHTS = {"15m": 0.25, "1h": 0.35, "4h": 0.40}

st.set_page_config(page_title="Gate AI Trading Intelligence V3.2", page_icon="📊", layout="wide")
st.title("📊 GATE AI TRADING INTELLIGENCE — V3.2")
st.caption("Gate.io USDT scanner • Early signal • Bull/Bear confirmation • ATR scenario levels • EMA • RSI • MACD • ADX • Volume • 15m / 1h / 4h")
st.warning("Alat bantu analisis berbasis aturan teknikal, bukan penasihat keuangan atau prediksi pasti. Tidak ada transaksi otomatis. Sinyal dapat salah dan seluruh modal berisiko.")

@st.cache_data(ttl=45, show_spinner=False)
def fetch_tickers():
    r = requests.get(f"{API}/spot/tickers", headers=HEADERS, timeout=25)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or not data:
        raise ValueError("Ticker Gate.io kosong atau format tidak sesuai.")
    d = pd.DataFrame(data)
    for c in ["last", "change_percentage", "quote_volume"]:
        d[c + "_n"] = pd.to_numeric(d[c], errors="coerce")
    d = d[d.currency_pair.astype(str).str.endswith("_USDT")]
    d = d.dropna(subset=["last_n", "quote_volume_n"])
    return d[d.last_n > 0].sort_values("quote_volume_n", ascending=False).reset_index(drop=True)

@st.cache_data(ttl=45, show_spinner=False)
def fetch_candles(pair, interval):
    r = requests.get(f"{API}/spot/candlesticks", params={"currency_pair": pair, "interval": interval, "limit": 240}, headers=HEADERS, timeout=25)
    r.raise_for_status()
    rows = r.json()
    if not rows:
        raise ValueError(f"Candle kosong: {pair} {interval}")
    d = pd.DataFrame(rows).iloc[:, :7]
    d.columns = ["time", "quote_volume", "close", "high", "low", "open", "base_volume"]
    for c in ["quote_volume", "close", "high", "low", "open", "base_volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["time"] = pd.to_datetime(pd.to_numeric(d.time, errors="coerce"), unit="s", utc=True)
    return d.dropna().sort_values("time").drop_duplicates("time").reset_index(drop=True)

def enrich(d):
    x = d.copy()
    c, h, l = x.close, x.high, x.low
    for n in [9, 21, 50, 200]:
        x[f"ema{n}"] = c.ewm(span=n, adjust=False, min_periods=n).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi"] = 100 - 100 / (1 + rs)
    x.loc[(loss == 0) & (gain > 0), "rsi"] = 100
    x.loc[(loss == 0) & (gain == 0), "rsi"] = 50
    e12, e26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    x["macd"] = e12 - e26
    x["macd_signal"] = x.macd.ewm(span=9, adjust=False).mean()
    prev = c.shift(1)
    tr = pd.concat([h-l, (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    x["atr"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    up, down = h.diff(), -l.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=x.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=x.index)
    atr_safe = x.atr.replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_safe
    minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_safe
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    x["plus_di"], x["minus_di"] = plus_di, minus_di
    x["adx"] = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    x["vol_ma20"] = x.base_volume.rolling(20).mean()
    return x

def assess(frame):
    r = frame.iloc[-1]
    score, reasons = 0.0, []
    def add(condition, points, positive, negative):
        nonlocal score
        if condition:
            score += points
            reasons.append(positive)
        else:
            score -= points
            reasons.append(negative)
    add(r.ema9 > r.ema21, 1.0, "EMA9 > EMA21", "EMA9 ≤ EMA21")
    if pd.notna(r.ema200):
        add(r.close > r.ema200, 1.0, "Harga di atas EMA200", "Harga di bawah EMA200")
    add(r.macd > r.macd_signal, 1.0, "MACD di atas signal", "MACD di bawah signal")
    if pd.notna(r.rsi):
        if 52 <= r.rsi < 70: score += 0.75; reasons.append(f"RSI {r.rsi:.1f}: momentum bullish")
        elif 30 < r.rsi <= 48: score -= 0.75; reasons.append(f"RSI {r.rsi:.1f}: momentum bearish")
        elif r.rsi >= 70: reasons.append(f"RSI {r.rsi:.1f}: tinggi, waspadai jenuh beli")
        elif r.rsi <= 30: reasons.append(f"RSI {r.rsi:.1f}: rendah, tunggu konfirmasi")
    if pd.notna(r.adx):
        if r.adx >= 20:
            if r.plus_di > r.minus_di: score += 0.75; reasons.append(f"ADX {r.adx:.1f}: tren naik cukup kuat (+DI > -DI)")
            elif r.minus_di > r.plus_di: score -= 0.75; reasons.append(f"ADX {r.adx:.1f}: tren turun cukup kuat (-DI > +DI)")
        else: reasons.append(f"ADX {r.adx:.1f}: tren belum kuat")
    if pd.notna(r.vol_ma20) and r.vol_ma20 > 0 and r.base_volume > r.vol_ma20:
        score += 0.5 if score >= 0 else -0.5
        reasons.append("Volume di atas rata-rata 20 candle")
    trend = "Bullish" if score >= 2 else "Bearish" if score <= -2 else "Netral"
    return {"score": score, "trend": trend, "rsi": r.rsi, "adx": r.adx, "atr": r.atr, "last": r.close, "reasons": reasons}

def decide(res):
    raw = sum(res[k]["score"] * WEIGHTS[k] for k in TFS)
    confidence = int(np.clip(round(50 + raw / 5.0 * 50), 0, 100))
    if raw >= 1.35 and res["1h"]["trend"] == res["4h"]["trend"] == "Bullish": signal = "BULLISH TERKONFIRMASI"
    elif raw <= -1.35 and res["1h"]["trend"] == res["4h"]["trend"] == "Bearish": signal = "BEARISH TERKONFIRMASI"
    else: signal = "WAIT / BELUM TERKONFIRMASI"
    return signal, confidence, raw

def early_signal(frame):
    """Deteksi momentum awal pada 15m; belum merupakan konfirmasi tren."""
    x = frame.dropna(subset=["ema9", "ema21", "macd", "macd_signal", "rsi"])
    if len(x) < 3:
        return "DATA KURANG"
    a, b = x.iloc[-1], x.iloc[-2]
    hist_now = a.macd - a.macd_signal
    hist_prev = b.macd - b.macd_signal
    if b.ema9 <= b.ema21 and a.ema9 > a.ema21 and hist_now > hist_prev and a.rsi >= 50:
        return "POTENSI BULLISH AWAL"
    if b.ema9 >= b.ema21 and a.ema9 < a.ema21 and hist_now < hist_prev and a.rsi <= 50:
        return "POTENSI BEARISH AWAL"
    if a.ema9 > a.ema21 and hist_now > hist_prev and 50 <= a.rsi < 65:
        return "MOMENTUM BULLISH MENINGKAT"
    if a.ema9 < a.ema21 and hist_now < hist_prev and 35 < a.rsi <= 50:
        return "MOMENTUM BEARISH MENINGKAT"
    return "BELUM ADA PEMICU AWAL"


def atr_scenario(price, atr, signal, sl_multiple=1.5, reward_risk=2.0):
    """Level skenario berbasis ATR, bukan instruksi transaksi."""
    if not all(pd.notna(v) and np.isfinite(v) and v > 0 for v in [price, atr]):
        return {"Entry referensi": np.nan, "Stop skenario": np.nan, "Target skenario": np.nan, "Risk/Reward": np.nan}
    if signal == "BULLISH TERKONFIRMASI":
        stop = price - sl_multiple * atr
        target = price + reward_risk * sl_multiple * atr
    elif signal == "BEARISH TERKONFIRMASI":
        stop = price + sl_multiple * atr
        target = price - reward_risk * sl_multiple * atr
    else:
        return {"Entry referensi": price, "Stop skenario": np.nan, "Target skenario": np.nan, "Risk/Reward": np.nan}
    return {"Entry referensi": price, "Stop skenario": stop, "Target skenario": target, "Risk/Reward": reward_risk}


def scan_one(pair):
    try:
        frames = {tf: enrich(fetch_candles(pair, tf)) for tf in TFS}
        if any(len(frames[tf]) < 205 for tf in TFS):
            return {"Pair": pair, "Sinyal": "DATA KURANG", "Skor": np.nan, "Konfirmasi": np.nan, "Harga": frames["1h"].iloc[-1].close, "RSI 1h": np.nan, "ADX 1h": np.nan, "Volume 24h USDT": np.nan}
        res = {tf: assess(frames[tf]) for tf in TFS}
        signal, conf, raw = decide(res)
        early = early_signal(frames["15m"])
        levels = atr_scenario(res["1h"]["last"], res["1h"]["atr"], signal)
        return {"Pair": pair, "Sinyal": signal, "Sinyal awal 15m": early,
                "Skor": round(raw, 2), "Konfirmasi": conf, "Harga": res["1h"]["last"],
                "RSI 1h": round(res["1h"]["rsi"], 1) if pd.notna(res["1h"]["rsi"]) else np.nan,
                "ADX 1h": round(res["1h"]["adx"], 1) if pd.notna(res["1h"]["adx"]) else np.nan,
                **levels, "Volume 24h USDT": np.nan}
    except Exception as e:
        return {"Pair": pair, "Sinyal": "GAGAL: " + str(e)[:65], "Skor": np.nan, "Konfirmasi": np.nan, "Harga": np.nan, "RSI 1h": np.nan, "ADX 1h": np.nan, "Volume 24h USDT": np.nan}

with st.sidebar:
    st.header("Pengaturan scanner")
    scan_limit = st.select_slider("Jumlah pair dipindai", options=[10, 20, 30, 50, 100], value=30)
    min_volume = st.number_input("Volume minimum 24 jam (USDT)", min_value=0, value=100000, step=50000)
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear(); st.rerun()
    st.caption("Data berasal dari Gate.io Public API. API key tidak diperlukan.")

try:
    tickers = fetch_tickers()
    eligible = tickers[tickers.quote_volume_n >= min_volume].head(scan_limit).copy()
    eligible_pairs = eligible.currency_pair.tolist()
    st.subheader(f"Market Scanner — {len(eligible_pairs)} pair dipilih")
    st.caption("Pair disaring berdasarkan volume 24 jam, lalu dianalisis pada 15m, 1h, dan 4h. Pemindaian 100 pair dapat membutuhkan waktu lebih lama.")
    if not eligible_pairs:
        st.info("Tidak ada pair yang memenuhi volume minimum. Turunkan batas volume di sidebar.")
        st.stop()
    if st.button("▶ Jalankan pemindaian Bull/Bear", type="primary") or "scan_results" not in st.session_state:
        progress = st.progress(0, text="Memulai pemindaian...")
        output = []
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(scan_one, p): p for p in eligible_pairs}
            done = 0
            for f in as_completed(futures):
                output.append(f.result()); done += 1
                progress.progress(done / len(futures), text=f"Memindai {done}/{len(futures)} pair")
        df = pd.DataFrame(output)
        vol_map = eligible.set_index("currency_pair").quote_volume_n.to_dict()
        df["Volume 24h USDT"] = df.Pair.map(vol_map)
        st.session_state.scan_results = df
        st.session_state.scan_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    result = st.session_state.scan_results.copy()
    order = {"BULLISH TERKONFIRMASI": 0, "BEARISH TERKONFIRMASI": 1, "WAIT / BELUM TERKONFIRMASI": 2, "DATA KURANG": 3}
    result["_order"] = result.Sinyal.map(order).fillna(4)
    result = result.sort_values(["_order", "Konfirmasi", "Volume 24h USDT"], ascending=[True, False, False]).drop(columns="_order")
    st.subheader("Hasil pemindaian")
    bull = int((result.Sinyal == "BULLISH TERKONFIRMASI").sum())
    bear = int((result.Sinyal == "BEARISH TERKONFIRMASI").sum())
    wait = int((result.Sinyal == "WAIT / BELUM TERKONFIRMASI").sum())
    a,b,c = st.columns(3); a.metric("Bullish", bull); b.metric("Bearish", bear); c.metric("Wait", wait)
    st.dataframe(result, use_container_width=True, hide_index=True)
    st.caption("Peringkat hanya mengurutkan sinyal teknikal, bukan jaminan keberhasilan. Data gagal/kosong tidak boleh dianggap sinyal.")
    st.caption("Sinyal awal 15m adalah peringatan momentum, bukan konfirmasi. Level ATR adalah simulasi matematis, bukan perintah transaksi.")
    st.caption("Waktu pemindaian: " + st.session_state.get("scan_time", "—"))

    st.divider(); st.header("Analisis detail satu pair")
    pairs = eligible_pairs
    default_pair = "AKE_USDT" if "AKE_USDT" in pairs else ("BTC_USDT" if "BTC_USDT" in pairs else pairs[0])
    pair = st.selectbox("Pilih pair", pairs, index=pairs.index(default_pair))
    frames = {tf: enrich(fetch_candles(pair, tf)) for tf in TFS}
    res = {tf: assess(frames[tf]) for tf in TFS}
    signal, conf, raw = decide(res)
    early = early_signal(frames["15m"])
    levels = atr_scenario(res["1h"]["last"], res["1h"]["atr"], signal)
    st.subheader(f"Kesimpulan: {pair}")
    st.info(f"Sinyal awal 15m: **{early}**")
    c1,c2,c3 = st.columns(3); c1.metric("Harga referensi 1h", f"{res['1h']['last']:.8g} USDT"); c2.metric("Skor konfirmasi", f"{conf}/100"); c3.metric("Sinyal", signal)
    cols = st.columns(3)
    for col, tf in zip(cols, TFS):
        with col:
            col.markdown(f"### {tf} — {res[tf]['trend']}")
            col.metric("RSI 14", "—" if pd.isna(res[tf]["rsi"]) else f"{res[tf]['rsi']:.1f}")
            col.metric("ADX 14", "—" if pd.isna(res[tf]["adx"]) else f"{res[tf]['adx']:.1f}")
            for reason in res[tf]["reasons"]: col.write("• " + reason)
    chart_tf = st.selectbox("Timeframe grafik", TFS, index=1)
    chart = frames[chart_tf].tail(150)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=chart.time, open=chart.open, high=chart.high, low=chart.low, close=chart.close, name="Candlestick"))
    for n in [9,21,200]: fig.add_trace(go.Scatter(x=chart.time, y=chart[f"ema{n}"], name=f"EMA{n}", line=dict(width=1.2)))
    fig.update_layout(height=480, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=25,b=10))
    st.plotly_chart(fig, use_container_width=True)
    atr = res["1h"]["atr"]
    if pd.notna(atr) and atr > 0:
        st.subheader("Volatilitas historis")
        x,y = st.columns(2); x.metric("ATR 14 (1h)", f"{atr:.8g}"); y.metric("ATR relatif", f"{atr/res['1h']['last']*100:.2f}%")
        st.caption("ATR adalah ukuran volatilitas historis per candle. Level berikut hanya simulasi jika sinyal terkonfirmasi.")
        q1, q2, q3 = st.columns(3)
        q1.metric("Entry referensi", f"{levels['Entry referensi']:.8g}" if pd.notna(levels["Entry referensi"]) else "—")
        q2.metric("Stop skenario", f"{levels['Stop skenario']:.8g}" if pd.notna(levels["Stop skenario"]) else "—")
        q3.metric("Target skenario", f"{levels['Target skenario']:.8g}" if pd.notna(levels["Target skenario"]) else "—")
    with st.expander("Data candle terbaru"):
        d = frames[chart_tf][["time","open","high","low","close","base_volume","ema9","ema21","ema200","rsi","adx","atr"]].tail(20).copy()
        d["time"] = d.time.dt.strftime("%Y-%m-%d %H:%M UTC")
        st.dataframe(d.sort_values("time", ascending=False), use_container_width=True, hide_index=True)
    st.caption("Bobot skor: 15m 25%, 1h 35%, 4h 40%. Sinyal bull/bear wajib searah pada 1h dan 4h serta melewati ambang skor. Scanner hanya membaca API publik dan tidak mengirim order.")
except requests.RequestException as e:
    st.error("Koneksi ke Gate.io gagal. Coba refresh beberapa saat lagi."); st.code(str(e))
except Exception as e:
    st.error("Gagal memproses data."); st.code(f"{type(e).__name__}: {e}")
