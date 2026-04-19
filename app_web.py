# app_web.py - HyperBot Trading Strategy Tester — Refonte UI dark terminal
"""
🌐 HYPERBOT WEB APP - Version avec Auto-Clean
Application web permettant d'uploader dynamiquement:
- Le fichier de stratégie (hyperbot_core.py)
- Les données historiques (CSV) - Nettoyage automatique inclus
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import sys
import os
import tempfile
import importlib.util
from datetime import datetime
import traceback
import re

st.set_page_config(
    page_title="HyperBot · Strategy Tester",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS GLOBAL DARK TERMINAL ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --bg:      #050e0a;
    --bg2:     #071410;
    --bg3:     #0a1e16;
    --panel:   #091912;
    --border:  #0d2e1e;
    --border2: #154030;
    --green:   #00e676;
    --green2:  #00c853;
    --red:     #ff5252;
    --yellow:  #ffb300;
    --blue:    #40c4ff;
    --muted:   #2a5a3a;
    --sub:     #153020;
    --font:    'JetBrains Mono', monospace;
}

html, body, [data-testid="stApp"] {
    background: var(--bg) !important;
    font-family: var(--font) !important;
    color: #a8d8b8 !important;
}
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
    color: #7ab88a !important;
    font-family: var(--font) !important;
    font-size: 12px !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--green) !important;
    font-size: 10px !important;
    letter-spacing: .12em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 5px !important;
    margin: 12px 0 8px !important;
}

/* Inputs */
[data-testid="stFileUploader"] {
    background: var(--bg3) !important;
    border: 1px dashed var(--border2) !important;
    border-radius: 3px !important;
}
[data-testid="stFileUploader"] * { font-size: 11px !important; }
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background: var(--bg3) !important;
    border: 1px solid var(--border2) !important;
    color: var(--green) !important;
    font-family: var(--font) !important;
    font-size: 12px !important;
    border-radius: 2px !important;
}
div[data-baseweb="select"] {
    background: var(--bg3) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 2px !important;
}
div[data-baseweb="select"] * { color: #7ab88a !important; font-size: 12px !important; }

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: var(--bg2) !important;
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stTabs"] button {
    color: var(--muted) !important;
    font-family: var(--font) !important;
    font-size: 11px !important;
    letter-spacing: .06em !important;
    background: transparent !important;
    border: none !important;
    padding: 8px 16px !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--green) !important;
    border-bottom: 2px solid var(--green) !important;
}

/* Buttons */
[data-testid="stButton"] button {
    background: transparent !important;
    border: 1px solid var(--green) !important;
    color: var(--green) !important;
    font-family: var(--font) !important;
    font-size: 11px !important;
    border-radius: 2px !important;
    letter-spacing: .06em !important;
}
[data-testid="stButton"] button:hover {
    background: var(--green) !important;
    color: var(--bg) !important;
}
button[kind="primary"] {
    background: var(--yellow) !important;
    border-color: var(--yellow) !important;
    color: #000 !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    letter-spacing: .08em !important;
}

/* Slider */
[data-testid="stSlider"] { filter: hue-rotate(100deg) saturate(1.5) !important; }

/* Checkbox */
[data-testid="stCheckbox"] * { color: #7ab88a !important; font-size: 11px !important; }

/* Metrics */
[data-testid="stMetric"] {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    padding: 10px 14px !important;
}
[data-testid="stMetricLabel"] { font-size: 10px !important; color: var(--muted) !important; }
[data-testid="stMetricValue"] { font-size: 18px !important; color: #c8f0d8 !important; }
[data-testid="stMetricDelta"] { font-size: 11px !important; }

/* Expander */
[data-testid="stExpander"] {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
}
[data-testid="stExpander"] summary { font-size: 11px !important; color: #4a8a5a !important; }

/* DataFrame */
[data-testid="stDataFrame"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
}

/* Line chart */
[data-testid="stVegaLiteChart"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
}

/* Code blocks */
[data-testid="stCode"] {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
}

/* Alerts */
[data-testid="stAlert"] {
    background: var(--bg3) !important;
    border-radius: 2px !important;
    font-size: 11px !important;
}

hr { border-color: var(--border) !important; margin: 8px 0 !important; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--sub); border-radius: 2px; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none !important; }
.block-container { padding-top: 0 !important; max-width: 100% !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TOUTE LA LOGIQUE BACKEND — INCHANGÉE
# ═══════════════════════════════════════════════════════════════════════════════

def auto_clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    log_messages = []
    try:
        log_messages.append("🔍 Analyse de la structure...")
        first_row = df.iloc[0]
        has_headers = True
        if all(isinstance(col, int) for col in df.columns):
            has_headers = False
            log_messages.append("⚠️ CSV SANS ENTÊTES détecté (colonnes numériques)")
        elif df.columns[0] and ('-' in str(df.columns[0]) or '/' in str(df.columns[0]) or ':' in str(df.columns[0])):
            try:
                pd.to_datetime(str(df.columns[0]))
                has_headers = False
                log_messages.append("⚠️ CSV SANS ENTÊTES détecté (première ligne = date)")
            except:
                pass
        if not has_headers:
            num_cols = len(df.columns)
            if num_cols == 6:
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                log_messages.append("✅ Entêtes ajoutées: timestamp, open, high, low, close, volume")
            elif num_cols == 5:
                df.columns = ['timestamp', 'open', 'high', 'low', 'close']
                df['volume'] = 1000000
                log_messages.append("✅ Entêtes ajoutées: timestamp, open, high, low, close (volume ajouté)")
            elif num_cols == 7:
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'adj close', 'volume']
                log_messages.append("✅ Entêtes ajoutées: timestamp, open, high, low, close, adj close, volume")
            else:
                raise ValueError(f"❌ Nombre de colonnes non supporté: {num_cols}. Attendu: 5, 6 ou 7 colonnes.")
        column_mapping = {
            "date": "timestamp", "time": "timestamp", "datetime": "timestamp",
            "price": "close", "last": "close", "close price": "close",
            "vol.": "volume", "vol": "volume",
            "adj close": "adj close", "adjusted": "adj close",
        }
        df.columns = df.columns.str.strip().str.lower()
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        log_messages.append(f"✅ Colonnes finales: {list(df.columns)}")
        if "timestamp" not in df.columns:
            raise ValueError("❌ Aucune colonne de date/temps trouvée!")
        log_messages.append("📅 Conversion des dates...")
        sample_date = str(df["timestamp"].iloc[0]).strip()
        if "-" in sample_date and (":" in sample_date or " " in sample_date):
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M")
                log_messages.append("✅ Format détecté: YYYY-MM-DD HH:MM")
            except:
                try:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
                    log_messages.append("✅ Format détecté: YYYY-MM-DD HH:MM:SS")
                except:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    log_messages.append("✅ Format détecté automatiquement")
        elif "/" in sample_date:
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%Y")
                log_messages.append("✅ Format détecté: MM/DD/YYYY")
            except:
                try:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d/%m/%Y")
                    log_messages.append("✅ Format détecté: DD/MM/YYYY")
                except:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    log_messages.append("✅ Format détecté automatiquement")
        elif "-" in sample_date:
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d")
                log_messages.append("✅ Format détecté: YYYY-MM-DD")
            except:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                log_messages.append("✅ Format détecté automatiquement")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            log_messages.append("✅ Format de date deviné par pandas")
        log_messages.append(f"📆 Période: {df['timestamp'].min()} → {df['timestamp'].max()}")
        log_messages.append("💰 Nettoyage des prix...")
        for col in ["close", "open", "high", "low"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(",", "").str.strip()
                df[col] = pd.to_numeric(df[col], errors="coerce")
        log_messages.append(f"✅ Prix nettoyés (moyenne close: {df['close'].mean():.5f})")
        if "volume" in df.columns:
            log_messages.append("📊 Conversion du volume...")
            def parse_volume(v):
                if pd.isna(v): return 0.0
                if isinstance(v, (int, float)): return float(v)
                v_str = str(v).strip().upper().replace(",", "").replace(" ", "")
                if v_str.endswith("K"):   return float(v_str[:-1]) * 1_000
                elif v_str.endswith("M"): return float(v_str[:-1]) * 1_000_000
                elif v_str.endswith("B"): return float(v_str[:-1]) * 1_000_000_000
                else:
                    try: return float(v_str)
                    except: return 0.0
            df["volume"] = df["volume"].apply(parse_volume)
            log_messages.append(f"✅ Volume traité (moyenne: {df['volume'].mean():.0f})")
        else:
            df["volume"] = 1000000
            log_messages.append("⚠️ Pas de volume → valeur par défaut ajoutée")
        if "adj close" not in df.columns:
            df["adj close"] = df["close"]
        final_columns = ["timestamp", "open", "high", "low", "close", "adj close", "volume"]
        missing = [c for c in ["timestamp", "open", "high", "low", "close"] if c not in df.columns]
        if missing:
            raise ValueError(f"❌ Colonnes manquantes après nettoyage: {missing}")
        df = df[final_columns]
        log_messages.append("🧹 Nettoyage final...")
        before_clean = len(df)
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)
        after_clean = len(df)
        if before_clean != after_clean:
            log_messages.append(f"⚠️ {before_clean - after_clean} lignes avec NaN supprimées")
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df[(df["high"] >= df["low"]) & (df["high"] >= df["open"]) &
                (df["high"] >= df["close"]) & (df["low"] <= df["open"]) &
                (df["low"] <= df["close"])]
        log_messages.append(f"✅ Données nettoyées: {len(df)} bougies valides")
        return df, "\n".join(log_messages)
    except Exception as e:
        error_msg = f"❌ Erreur lors du nettoyage: {str(e)}\n{traceback.format_exc()}"
        log_messages.append(error_msg)
        return None, "\n".join(log_messages)


def load_strategy_from_upload(uploaded_file):
    try:
        content = uploaded_file.read().decode('utf-8')
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        spec = importlib.util.spec_from_file_location("hyperbot_core", tmp_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules['hyperbot_core'] = module
        spec.loader.exec_module(module)
        os.unlink(tmp_path)
        return module, None
    except Exception as e:
        return None, str(e)


@st.cache_data
def load_csv_safely(uploaded_file) -> tuple[pd.DataFrame, str]:
    try:
        content = uploaded_file.read().decode("utf-8")
        lines = content.split('\n')[:5]
        separator = ','
        sep_name = "Virgule"
        for line in lines:
            if not line.strip(): continue
            tab_count = line.count('\t')
            comma_count = line.count(',')
            semicolon_count = line.count(';')
            if tab_count > comma_count and tab_count > semicolon_count and tab_count >= 4:
                separator = '\t'; sep_name = "TAB"; break
            elif semicolon_count > comma_count and semicolon_count >= 4:
                separator = ';'; sep_name = "Point-virgule"; break
            elif comma_count >= 4:
                separator = ','; sep_name = "Virgule"; break
        df = pd.read_csv(io.StringIO(content), sep=separator, header=None, low_memory=False)
        cleaned_df, log = auto_clean_data(df)
        if cleaned_df is None:
            return None, log
        final_log = f"🔍 Séparateur détecté: {sep_name} ({df.shape[1]} colonnes)\n{log}"
        return cleaned_df, final_log
    except Exception as e:
        return None, f"❌ Erreur lors du chargement CSV: {e}\n{traceback.format_exc()}"


def detect_asset_type(df: pd.DataFrame) -> str:
    if df is None or df.empty: return "unknown"
    avg_price = df['close'].mean()
    volatility = ((df['high'] - df['low']).mean() / df['close'].mean()) * 100
    if avg_price > 500: return "crypto"
    elif avg_price < 10: return "forex"
    else: return "crypto" if volatility > 2 else "forex"


# ═══════════════════════════════════════════════════════════════════════════════
#  UI PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════════════

def main():

    # ── HEADER BAR ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        display:flex; align-items:center; gap:0;
        background:#071410; border-bottom:1px solid #0d2e1e;
        font-family:'JetBrains Mono',monospace;
        font-size:12px; padding:0;
    ">
      <div style="padding:12px 20px;border-right:1px solid #0d2e1e;display:flex;align-items:center;gap:8px">
        <span style="color:#ffb300;font-size:14px">⚡</span>
        <span style="color:#00e676;font-weight:700;letter-spacing:.04em">HyperBot</span>
        <span style="color:#2a5a3a">·</span>
        <span style="color:#4a8a5a">Strategy Tester</span>
      </div>
      <div style="display:flex;gap:0">
        <div style="padding:12px 18px;border-right:1px solid #0d2e1e;color:#00e676;font-weight:700;border-bottom:2px solid #00e676">BACKTEST</div>
        <div style="padding:12px 18px;border-right:1px solid #0d2e1e;color:#2a5a3a">LIVE</div>
        <div style="padding:12px 18px;border-right:1px solid #0d2e1e;color:#2a5a3a">STRATÉGIES</div>
        <div style="padding:12px 18px;border-right:1px solid #0d2e1e;color:#2a5a3a">ANALYTICS</div>
        <div style="padding:12px 18px;color:#2a5a3a">LOGS</div>
      </div>
      <div style="margin-left:auto;display:flex;align-items:center;gap:16px;padding-right:20px">
        <span style="color:#4a8a5a">● Backtest mode</span>
        <span style="color:#4a8a5a">● Data: Binance API</span>
        <span style="color:#2a5a3a;font-size:11px">2024-01-01 → 2024-12-31</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:

        # ── Stratégies list (visuel screenshot) ───────────────────────────────
        st.markdown("""
        <div style="font-size:9px;color:#2a5a3a;letter-spacing:.12em;
                    text-transform:uppercase;margin-bottom:8px">Stratégies</div>
        """, unsafe_allow_html=True)

        strategies_demo = [
            ("RSI + EMA",     "+12.4%", True),
            ("MACD Cross",    "+8.2%",  False),
            ("Bollinger BB",  "-3.1%",  False),
            ("Buy & Hold",    "+6.8%",  False),
        ]
        for name, perf, active in strategies_demo:
            color = "#00e676" if "+" in perf else "#ff5252"
            bg    = "#0a1e16" if active else "transparent"
            border= "1px solid #0d2e1e" if active else "1px solid transparent"
            st.markdown(f"""
            <div style="
                display:flex;justify-content:space-between;align-items:center;
                padding:7px 10px; margin-bottom:2px;
                background:{bg}; border:{border}; border-radius:2px;
                font-family:'JetBrains Mono',monospace; font-size:11px;
                cursor:pointer;
            ">
              <span style="color:{'#00e676' if active else '#4a8a5a'}">{name}</span>
              <span style="color:{color};font-weight:700;font-size:10px">{perf}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Paramètres (visuel screenshot) ────────────────────────────────────
        st.markdown("""
        <div style="font-size:9px;color:#2a5a3a;letter-spacing:.12em;
                    text-transform:uppercase;margin-bottom:8px">Paramètres</div>
        """, unsafe_allow_html=True)

        rsi_period = st.number_input("RSI Period",  value=14, min_value=2)
        ema_fast   = st.number_input("EMA Fast",    value=9,  min_value=2)
        ema_slow   = st.number_input("EMA Slow",    value=21, min_value=2)

        st.markdown("""
        <div style="display:flex;justify-content:space-between;
                    font-family:'JetBrains Mono',monospace;font-size:11px;
                    color:#4a8a5a;padding:4px 0">
          <span>Stop Loss</span><span style="color:#ff5252">2%</span>
        </div>
        <div style="display:flex;justify-content:space-between;
                    font-family:'JetBrains Mono',monospace;font-size:11px;
                    color:#4a8a5a;padding:4px 0">
          <span>Take Profit</span><span style="color:#00e676">4%</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Période ───────────────────────────────────────────────────────────
        st.markdown("""
        <div style="font-size:9px;color:#2a5a3a;letter-spacing:.12em;
                    text-transform:uppercase;margin-bottom:8px">Période</div>
        <div style="display:flex;justify-content:space-between;
                    font-family:'JetBrains Mono',monospace;font-size:11px;
                    color:#4a8a5a;padding:4px 0">
          <span>Début</span><span style="color:#a8d8b8">01/01/24</span>
        </div>
        <div style="display:flex;justify-content:space-between;
                    font-family:'JetBrains Mono',monospace;font-size:11px;
                    color:#4a8a5a;padding:4px 0">
          <span>Fin</span><span style="color:#a8d8b8">31/12/24</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Upload fichiers ───────────────────────────────────────────────────
        st.markdown("### 📁 Upload")
        st.markdown('<div style="font-size:9px;color:#2a5a3a;margin-bottom:4px">1 · Stratégie Python</div>', unsafe_allow_html=True)
        strategy_file = st.file_uploader("hyperbot_core.py", type=["py"],
                                          label_visibility="collapsed")

        st.markdown('<div style="font-size:9px;color:#2a5a3a;margin:8px 0 4px">2 · Données CSV OHLCV</div>', unsafe_allow_html=True)
        data_file = st.file_uploader("CSV OHLCV", type=["csv"],
                                      label_visibility="collapsed")

        with st.expander("ℹ️ Formats supportés"):
            st.markdown("""
**Sources:** Investing.com · IPMS · Yahoo Finance · TradingView · HistData · CryptoDataDownload

**Colonnes:** Date/Time · Open High Low Close · Volume (K/M/B)

✨ CSV sans entêtes supporté
            """)

    # ── ZONE PRINCIPALE ───────────────────────────────────────────────────────
    if not strategy_file or not data_file:
        # ── État vide — onboarding visuel ─────────────────────────────────────
        col_chart, col_config = st.columns([3, 1], gap="small")

        with col_chart:
            # Chart démo
            st.markdown("""
            <div style="
                padding:14px 20px;
                background:#071410; border-bottom:1px solid #0d2e1e;
                display:flex;align-items:center;gap:16px;
                font-family:'JetBrains Mono',monospace;
            ">
              <span style="font-size:16px;font-weight:700;color:#a8d8b8">BTC/USDT</span>
              <span style="font-size:20px;font-weight:700;color:#a8d8b8">$67,430</span>
              <span style="color:#00e676;font-size:12px">+2.34%</span>
              <span style="color:#4a8a5a;font-size:11px">Stratégie: RSI+EMA · 4H</span>
              <div style="margin-left:auto;display:flex;gap:4px">
                <span style="padding:4px 10px;border:1px solid #0d2e1e;border-radius:2px;font-size:11px;color:#4a8a5a">1H</span>
                <span style="padding:4px 10px;background:#0a1e16;border:1px solid #00e676;border-radius:2px;font-size:11px;color:#00e676">4H</span>
                <span style="padding:4px 10px;border:1px solid #0d2e1e;border-radius:2px;font-size:11px;color:#4a8a5a">1D</span>
                <span style="padding:4px 10px;border:1px solid #0d2e1e;border-radius:2px;font-size:11px;color:#4a8a5a">1W</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Demo equity curve avec SVG
            st.markdown("""
            <div style="
                height:340px; background:#050e0a;
                border:1px solid #0d2e1e; border-radius:0;
                position:relative; overflow:hidden;
                font-family:'JetBrains Mono',monospace;
            ">
              <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
                <!-- grid -->
                <defs>
                  <pattern id="grid" width="60" height="40" patternUnits="userSpaceOnUse">
                    <path d="M 60 0 L 0 0 0 40" fill="none" stroke="#091912" stroke-width="0.8"/>
                  </pattern>
                  <linearGradient id="chartFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="#00e676" stop-opacity="0.15"/>
                    <stop offset="100%" stop-color="#00e676" stop-opacity="0.01"/>
                  </linearGradient>
                </defs>
                <rect width="100%" height="100%" fill="url(#grid)"/>

                <!-- EMA dashed lines -->
                <polyline points="0,300 100,285 200,260 300,240 400,220 500,200 600,185 700,165 800,150 900,135 1000,115"
                  fill="none" stroke="#ffb300" stroke-width="1" stroke-dasharray="4 4" opacity="0.5"/>
                <polyline points="0,290 100,278 200,255 300,238 400,218 500,198 600,183 700,162 800,148 900,132 1000,112"
                  fill="none" stroke="#40c4ff" stroke-width="1" stroke-dasharray="4 4" opacity="0.5"/>

                <!-- price fill area -->
                <polygon points="0,310 80,295 130,260 180,280 230,250 300,230 380,220 450,240 520,210 600,195 680,180 760,170 840,155 920,140 1000,120 1000,340 0,340"
                  fill="url(#chartFill)"/>

                <!-- price line -->
                <polyline points="0,310 80,295 130,260 180,280 230,250 300,230 380,220 450,240 520,210 600,195 680,180 760,170 840,155 920,140 1000,120"
                  fill="none" stroke="#00e676" stroke-width="1.5"/>

                <!-- BUY signals (up triangles) -->
                <polygon points="90,290 96,300 84,300" fill="#00e676" opacity="0.9"/>
                <polygon points="310,224 316,234 304,234" fill="#00e676" opacity="0.9"/>
                <polygon points="530,203 536,213 524,213" fill="#00e676" opacity="0.9"/>
                <polygon points="760,162 766,172 754,172" fill="#00e676" opacity="0.9"/>

                <!-- SELL signals (down triangles) -->
                <polygon points="190,274 196,264 184,264" fill="#ff5252" opacity="0.9"/>
                <polygon points="460,234 466,224 454,224" fill="#ff5252" opacity="0.9"/>

                <!-- target line -->
                <line x1="0" y1="55" x2="100%" y2="55"
                  stroke="#00e676" stroke-width="0.8" stroke-dasharray="3 3" opacity="0.3"/>
                <text x="8" y="50" font-family="JetBrains Mono,monospace" font-size="9" fill="#00e676" opacity="0.5">TP</text>

                <!-- price labels right -->
                <text x="calc(100% - 60px)" y="124" font-family="JetBrains Mono,monospace" font-size="9" fill="#4a8a5a">67,430</text>
                <text x="calc(100% - 60px)" y="175" font-family="JetBrains Mono,monospace" font-size="9" fill="#4a8a5a">62,000</text>
                <text x="calc(100% - 60px)" y="230" font-family="JetBrains Mono,monospace" font-size="9" fill="#4a8a5a">55,000</text>
                <text x="calc(100% - 60px)" y="310" font-family="JetBrains Mono,monospace" font-size="9" fill="#4a8a5a">45,000</text>

                <!-- placeholder text -->
                <text x="50%" y="50%" text-anchor="middle" dominant-baseline="central"
                  font-family="JetBrains Mono,monospace" font-size="13" fill="#154030">
                  Uploadez vos fichiers pour lancer le backtest
                </text>
              </svg>
            </div>
            """, unsafe_allow_html=True)

            # Bottom metrics bar (screenshot style)
            st.markdown("""
            <div style="
                display:grid; grid-template-columns:repeat(6,1fr);
                border:1px solid #0d2e1e; border-top:none;
                font-family:'JetBrains Mono',monospace;
            ">
            """ + "".join([
                f"""<div style="padding:10px 14px;border-right:{'1px solid #0d2e1e' if i<5 else 'none'}">
                  <div style="font-size:9px;color:#2a5a3a;letter-spacing:.08em;text-transform:uppercase">{label}</div>
                  <div style="font-size:15px;font-weight:700;color:{color}">{val}</div>
                </div>"""
                for i, (label, val, color) in enumerate([
                    ("P&L Total",    "+$1,247.83", "#00e676"),
                    ("Rendement",    "+12.47%",    "#00e676"),
                    ("Win Rate",     "68.4%",      "#a8d8b8"),
                    ("Trades",       "47",         "#a8d8b8"),
                    ("Max Drawdown", "-4.2%",      "#ff5252"),
                    ("Sharpe Ratio", "1.84",       "#40c4ff"),
                ])
            ]) + """
            </div>
            """, unsafe_allow_html=True)

        with col_config:
            # Config stratégie panel (screenshot right side)
            st.markdown("""
            <div style="
                background:#071410; border:1px solid #0d2e1e;
                border-radius:3px; overflow:hidden;
                font-family:'JetBrains Mono',monospace;
            ">
              <div style="padding:10px 14px;border-bottom:1px solid #0d2e1e;
                          font-size:10px;color:#2a5a3a;letter-spacing:.12em;text-transform:uppercase">
                Config Stratégie
              </div>
            """ + "".join([
                f"""<div style="display:flex;justify-content:space-between;align-items:center;
                        padding:8px 14px;border-bottom:1px solid #050e0a">
                  <span style="font-size:11px;color:#4a8a5a">{label}</span>
                  <span style="font-size:12px;font-weight:700;color:#a8d8b8;
                               background:#0a1e16;padding:2px 8px;border-radius:2px">{val}</span>
                </div>"""
                for label, val in [
                    ("Capital init.", "$10,000"),
                    ("RSI period",    "14"),
                    ("RSI oversold",  "30"),
                    ("RSI overbought","70"),
                    ("EMA cross",     "9/21"),
                    ("Position size", "10%"),
                ]
            ]) + """
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            # Run backtest button (yellow — screenshot)
            st.button("▶  RUN BACKTEST", use_container_width=True, type="primary",
                      disabled=True)

            # Derniers trades (screenshot)
            st.markdown("""
            <div style="
                margin-top:12px;
                background:#071410; border:1px solid #0d2e1e;
                border-radius:3px; overflow:hidden;
                font-family:'JetBrains Mono',monospace;
            ">
              <div style="padding:8px 14px;border-bottom:1px solid #0d2e1e;
                          font-size:10px;color:#2a5a3a;letter-spacing:.12em;text-transform:uppercase">
                Derniers Trades
              </div>
            """ + "".join([
                f"""<div style="display:flex;align-items:center;gap:8px;
                        padding:7px 12px;border-bottom:1px solid #050e0a">
                  <span style="
                      font-size:9px;font-weight:700;padding:2px 6px;border-radius:2px;
                      background:{'#002a10' if side=='BUY' else '#2a0000'};
                      color:{'#00e676' if side=='BUY' else '#ff5252'};
                      border:1px solid {'#00e676' if side=='BUY' else '#ff5252'}
                  ">{side}</span>
                  <div style="flex:1;min-width:0">
                    <div style="font-size:10px;color:#a8d8b8">BTC/USDT</div>
                    <div style="font-size:9px;color:#2a5a3a">{date}</div>
                  </div>
                  <div style="text-align:right">
                    <div style="font-size:11px;font-weight:700;color:{'#00e676' if '+' in pnl else '#ff5252'}">{pnl}</div>
                    <div style="font-size:9px;color:#2a5a3a">{pct}</div>
                  </div>
                </div>"""
                for side, date, pnl, pct in [
                    ("BUY",  "12/01 09:45", "+$184",  "+1.84%"),
                    ("SELL", "15/01 14:28", "-$67",   "-0.67%"),
                    ("BUY",  "22/01 08:15", "+$312",  "+3.12%"),
                    ("SELL", "28/01 10:50", "+$221",  "+2.21%"),
                    ("BUY",  "03/02 11:30", "-$45",   "-0.45%"),
                ]
            ]) + """
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="
            margin-top:16px; padding:12px 16px;
            background:#071410; border:1px solid #0d2e1e; border-radius:3px;
            font-family:'JetBrains Mono',monospace; font-size:11px; color:#2a5a3a;
            text-align:center;
        ">
          👈 Uploadez <span style="color:#00e676">hyperbot_core.py</span>
          et votre <span style="color:#00e676">CSV OHLCV</span> dans la sidebar pour lancer le backtest
        </div>
        """, unsafe_allow_html=True)
        return

    # ── FICHIERS CHARGÉS → PIPELINE COMPLET ──────────────────────────────────

    # Charger stratégie
    with st.spinner(""):
        module, error = load_strategy_from_upload(strategy_file)
        if error:
            st.markdown(f"""
            <div style="
                padding:12px 16px; background:#1a0a0a; border:1px solid #ff5252;
                border-left:3px solid #ff5252; border-radius:3px;
                font-family:'JetBrains Mono',monospace; font-size:11px; color:#ff8a8a;
            ">❌ Erreur stratégie: {error}</div>
            """, unsafe_allow_html=True)
            return
        st.markdown("""
        <div style="padding:8px 14px;background:#001a0e;border:1px solid #00e676;
                    border-left:3px solid #00e676;border-radius:3px;
                    font-family:'JetBrains Mono',monospace;font-size:11px;color:#00e676">
          ✔ Stratégie chargée</div>
        """, unsafe_allow_html=True)

    # Charger CSV
    with st.spinner(""):
        df, clean_log = load_csv_safely(data_file)
        with st.expander("📋 Log de nettoyage"):
            st.code(clean_log)
        if df is None:
            st.error("❌ Impossible de charger/nettoyer le CSV")
            return
        st.markdown(f"""
        <div style="padding:8px 14px;background:#001a0e;border:1px solid #00e676;
                    border-left:3px solid #00e676;border-radius:3px;
                    font-family:'JetBrains Mono',monospace;font-size:11px;color:#00e676">
          ✔ {len(df):,} bougies chargées et nettoyées</div>
        """, unsafe_allow_html=True)

    # Aperçu données
    with st.expander("👀 Aperçu des données nettoyées"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bougies",    f"{len(df):,}")
        c2.metric("Début",      df['timestamp'].iloc[0].strftime('%Y-%m-%d'))
        c3.metric("Fin",        df['timestamp'].iloc[-1].strftime('%Y-%m-%d'))
        c4.metric("Prix moyen", f"${df['close'].mean():.2f}")
        st.dataframe(df.head(10), use_container_width=True)

    asset_type = detect_asset_type(df)

    # ── LAYOUT principal: chart zone + config panel ───────────────────────────
    col_main, col_right = st.columns([3, 1], gap="small")

    with col_right:
        # CONFIG STRATÉGIE header
        st.markdown("""
        <div style="
            background:#071410; border:1px solid #0d2e1e; border-radius:3px;
            overflow:hidden; font-family:'JetBrains Mono',monospace;
        ">
          <div style="padding:10px 14px;border-bottom:1px solid #0d2e1e;
                      font-size:10px;color:#2a5a3a;letter-spacing:.12em;text-transform:uppercase">
            Config Stratégie
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_main:
        # Symbole header
        asset_type_sel = st.selectbox("Type d'actif", ["Auto-détection","Crypto","Forex"],
                                       label_visibility="collapsed")
        if asset_type_sel == "Crypto": asset_type = "crypto"
        elif asset_type_sel == "Forex": asset_type = "forex"

        pair_label = "BTC/USDT" if asset_type == "crypto" else "EUR/USD"
        mode_label = "CRYPTO" if asset_type == "crypto" else "FOREX"

        st.markdown(f"""
        <div style="
            padding:10px 16px; background:#071410;
            border:1px solid #0d2e1e;
            display:flex; align-items:center; gap:14px;
            font-family:'JetBrains Mono',monospace;
        ">
          <span style="font-size:16px;font-weight:700;color:#a8d8b8">{pair_label}</span>
          <span style="font-size:11px;color:#4a8a5a">Stratégie: RSI+EMA · 4H</span>
          <span style="margin-left:auto;font-size:10px;color:#2a5a3a;
                       background:#0a1e16;padding:3px 8px;border-radius:2px">{mode_label}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Paramètres en onglets ─────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Indicateurs", "💰 Risque", "🎯 TP / SL", "⚙ Avancé"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            ema_uf = st.number_input("EMA Ultra Fast", value=8  if asset_type=="forex" else 5,  min_value=3, max_value=20)
            ema_f  = st.number_input("EMA Fast",       value=21 if asset_type=="forex" else 13, min_value=5, max_value=50)
            ema_t  = st.number_input("EMA Trend",      value=55 if asset_type=="forex" else 34, min_value=20,max_value=200)
        with c2:
            rsi_len = st.number_input("RSI Length", value=14 if asset_type=="forex" else 9)
            atr_len = st.number_input("ATR Length", value=14 if asset_type=="forex" else 10)
        with c3:
            min_adx = st.number_input("Min ADX",  value=15 if asset_type=="forex" else 12, min_value=5, max_value=30)
            rsi_min = st.number_input("RSI Min",  value=25 if asset_type=="forex" else 30, min_value=10,max_value=40)
            rsi_max = st.number_input("RSI Max",  value=75 if asset_type=="forex" else 80, min_value=60,max_value=90)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            base_risk = st.slider("Risque de Base (%)", 1.0, 20.0, 3.0 if asset_type=="forex" else 10.0, 0.5)
            initial_capital = st.number_input("Capital Initial ($)", value=10000, step=1000)
        with c2:
            compound          = st.checkbox("Compound Agressif",  value=asset_type=="crypto")
            enable_protection = st.checkbox("Protection Sommets", value=False)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            sl_atr   = st.number_input("SL ATR",          value=2.0 if asset_type=="forex" else 1.2, step=0.1)
            tp_scalp = st.number_input("TP Scalp (ATR)",  value=1.5, step=0.1)
            tp_swing = st.number_input("TP Swing (ATR)",  value=3.0 if asset_type=="forex" else 4.0, step=0.5)
            tp_moon  = st.number_input("TP Moonshot (ATR)",value=6.0 if asset_type=="forex" else 12.0,step=0.5)
        with c2:
            scalp_percent  = st.slider("Scalp Percent",  10.0, 50.0, 25.0 if asset_type=="forex" else 20.0)
            breakeven_atr  = st.number_input("Breakeven ATR", value=0.8, step=0.1)

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            ultra_mode   = st.checkbox("Ultra Mode",          value=asset_type=="crypto")
            use_trailing = st.checkbox("Use Trailing Stop",   value=True)
            use_volume   = st.checkbox("Volume Filter",       value=False)
        with c2:
            commission = st.number_input("Commission (%)", value=0.002 if asset_type=="forex" else 0.1, format="%.3f")
            slippage   = st.number_input("Slippage (%)",   value=0.001 if asset_type=="forex" else 0.05,format="%.3f")

    # ── RUN BACKTEST button ───────────────────────────────────────────────────
    st.markdown("<div style='margin:12px 0'></div>", unsafe_allow_html=True)
    run = st.button("▶  RUN BACKTEST", type="primary", use_container_width=True)

    if run:
        try:
            config = module.StrategyConfig(
                ema_ultra_fast=ema_uf, ema_fast=ema_f, ema_trend=ema_t,
                rsi_len=rsi_len, atr_len=atr_len,
                base_risk=base_risk, initial_capital=initial_capital,
                compound_aggressive=compound,
                sl_atr_base=sl_atr, tp_scalp=tp_scalp, tp_swing=tp_swing,
                tp_moonshot=tp_moon, scalp_percent=scalp_percent,
                breakeven_atr=breakeven_atr,
                min_adx=min_adx, rsi_min=rsi_min, rsi_max=rsi_max,
                ultra_mode=ultra_mode, use_trailing=use_trailing,
                use_volume_filter=use_volume,
                enable_summit_protection=enable_protection,
                commission=commission, slippage=slippage,
            )
            strategy = module.HyperBotOptimized(config)

            with st.spinner(""):
                results_df = strategy.run_backtest(df.copy(), verbose=False)
                stats = strategy.get_statistics()

            # ── RÉSULTATS ─────────────────────────────────────────────────────
            st.markdown("""
            <div style="padding:8px 14px;background:#001a0e;border:1px solid #00e676;
                        border-left:3px solid #00e676;border-radius:3px;margin:8px 0;
                        font-family:'JetBrains Mono',monospace;font-size:11px;color:#00e676">
              ✔ Backtest terminé</div>
            """, unsafe_allow_html=True)

            if not stats or not isinstance(stats, dict):
                st.error("❌ Statistiques non disponibles")
                return

            default_stats = {
                'final_equity': initial_capital, 'total_return_pct': 0.0,
                'win_rate': 0.0, 'max_drawdown': 0.0, 'total_trades': 0,
                'profit_factor': 0.0, 'trades_per_month': 0.0,
                'avg_win': 0.0, 'avg_loss': 0.0,
                'duration_days': 0, 'monthly_return_pct': 0.0
            }
            stats = {**default_stats, **stats}

            # ── METRICS BAR (screenshot bottom bar style) ─────────────────────
            pnl_val  = stats['final_equity'] - initial_capital
            pnl_sign = "+" if pnl_val >= 0 else ""
            ret_sign = "+" if stats['total_return_pct'] >= 0 else ""
            pnl_color= "#00e676" if pnl_val >= 0 else "#ff5252"
            ret_color= "#00e676" if stats['total_return_pct'] >= 0 else "#ff5252"

            st.markdown(f"""
            <div style="
                display:grid; grid-template-columns:repeat(6,1fr);
                border:1px solid #0d2e1e; border-radius:3px; overflow:hidden;
                font-family:'JetBrains Mono',monospace; margin:8px 0;
            ">
            """ + "".join([
                f"""<div style="padding:10px 14px;border-right:{'1px solid #0d2e1e' if i<5 else 'none'};background:#071410">
                  <div style="font-size:9px;color:#2a5a3a;letter-spacing:.08em;text-transform:uppercase">{label}</div>
                  <div style="font-size:15px;font-weight:700;color:{color}">{val}</div>
                </div>"""
                for i, (label, val, color) in enumerate([
                    ("P&L Total",    f"{pnl_sign}${pnl_val:,.2f}",            pnl_color),
                    ("Rendement",    f"{ret_sign}{stats['total_return_pct']:.2f}%", ret_color),
                    ("Win Rate",     f"{stats['win_rate']:.1f}%",             "#a8d8b8"),
                    ("Trades",       f"{stats['total_trades']}",              "#a8d8b8"),
                    ("Max Drawdown", f"{stats['max_drawdown']:.2f}%",         "#ff5252"),
                    ("Profit Factor",f"{stats.get('profit_factor',0):.2f}",  "#40c4ff"),
                ])
            ]) + """</div>
            """, unsafe_allow_html=True)

            # Détails
            st.markdown("""
            <div style="font-size:9px;color:#2a5a3a;letter-spacing:.12em;
                        text-transform:uppercase;margin:12px 0 6px;
                        font-family:'JetBrains Mono',monospace">
              Détails des Performances</div>
            """, unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Trades/Mois",       f"{stats.get('trades_per_month',0):.1f}")
            c2.metric("Gain Moyen",        f"${stats.get('avg_win',0):,.2f}")
            c3.metric("Perte Moyenne",     f"${stats.get('avg_loss',0):,.2f}")
            c1.metric("Durée (jours)",     stats.get('duration_days',0))
            c2.metric("Rendement Mensuel", f"{stats.get('monthly_return_pct',0):.2f}%")

            # Pyramiding si dispo
            if 'max_pyramid_level' in stats or 'avg_pyramid_level' in stats:
                st.markdown("""
                <div style="font-size:9px;color:#2a5a3a;letter-spacing:.12em;
                            text-transform:uppercase;margin:12px 0 6px;
                            font-family:'JetBrains Mono',monospace">Pyramiding</div>
                """, unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                if 'max_pyramid_level' in stats:  c1.metric("Niveau Max", stats['max_pyramid_level'])
                if 'avg_pyramid_level' in stats:  c2.metric("Niveau Moyen", f"{stats['avg_pyramid_level']:.2f}")
                if 'total_entries' in stats:      c3.metric("Total Entrées", stats['total_entries'])

            # ── EQUITY CURVE ──────────────────────────────────────────────────
            st.markdown("""
            <div style="font-size:9px;color:#2a5a3a;letter-spacing:.12em;
                        text-transform:uppercase;margin:12px 0 6px;
                        font-family:'JetBrains Mono',monospace">Courbe d'Équité</div>
            """, unsafe_allow_html=True)

            if hasattr(strategy, 'equity_curve') and strategy.equity_curve:
                eq_df = pd.DataFrame({'Equity': strategy.equity_curve})
                st.line_chart(eq_df, color="#00e676", use_container_width=True)
            else:
                st.info("ℹ️ Aucune courbe d'équité disponible")

            # ── TRADES TABLE ──────────────────────────────────────────────────
            if hasattr(strategy, 'trades') and strategy.trades:
                st.markdown("""
                <div style="font-size:9px;color:#2a5a3a;letter-spacing:.12em;
                            text-transform:uppercase;margin:12px 0 6px;
                            font-family:'JetBrains Mono',monospace">Derniers Trades</div>
                """, unsafe_allow_html=True)

                trades_df = pd.DataFrame(strategy.trades)
                c1, c2, _ = st.columns([1, 1, 2])
                with c1:
                    filter_side = st.multiselect("Direction", options=trades_df['side'].unique(),
                                                  default=trades_df['side'].unique())
                with c2:
                    filter_result = st.multiselect("Résultat", options=['Gagnant','Perdant'],
                                                    default=['Gagnant','Perdant'])

                filtered = trades_df[
                    (trades_df['side'].isin(filter_side)) &
                    (trades_df['pnl'] > 0  if 'Gagnant' in filter_result else True) &
                    (trades_df['pnl'] <= 0 if 'Perdant' in filter_result else True)
                ]
                st.dataframe(filtered, use_container_width=True, height=380)

                csv = filtered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Télécharger les trades (CSV)", csv,
                    f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
            else:
                st.info("ℹ️ Aucun trade exécuté pendant le backtest")

            if stats['total_trades'] == 0:
                st.warning("⚠️ Aucun trade détecté — réduire min_adx, désactiver ultra_mode pour Forex")

        except Exception as e:
            st.error(f"❌ Erreur backtest: {str(e)}")
            with st.expander("🔍 Stack trace"):
                st.code(traceback.format_exc())
            st.markdown("**Colonnes:**"); st.write(list(df.columns))
            st.dataframe(df.head())


# ── FOOTER ────────────────────────────────────────────────────────────────────
def show_footer():
    st.markdown("""
    <div style="
        margin-top:16px; padding:10px 16px;
        display:flex; justify-content:space-between;
        border-top:1px solid #0d2e1e;
        font-family:'JetBrains Mono',monospace; font-size:10px; color:#1a4a2a;
    ">
      <span>✨ Auto-Clean · Multi-Source · Investing.com / IPMS / Yahoo</span>
      <span>HyperBot v2.1 · Trading Strategy Tester</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    show_footer()
