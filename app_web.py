# app_web.py - Application Web avec Nettoyage Automatique des Données
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

# Configuration de la page
st.set_page_config(
    page_title="🚀 HyperBot Web",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
#  FONCTION DE NETTOYAGE AUTOMATIQUE DES DONNÉES
# =========================================================
def auto_clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Nettoie automatiquement les données depuis différentes sources
    (Investing.com, IPMS, Yahoo Finance, etc.)
    
    Returns:
        (DataFrame nettoyé, message de log)
    """
    log_messages = []
    
    try:
        # === ÉTAPE 1: Détection et normalisation des colonnes ===
        log_messages.append("🔍 Détection du format des données...")
        
        # Mapper les noms de colonnes possibles
        column_mapping = {
            # Timestamps
            "date": "timestamp",
            "time": "timestamp",
            "datetime": "timestamp",
            
            # Prix
            "price": "close",
            "last": "close",
            "close price": "close",
            
            # Volume
            "vol.": "volume",
            "vol": "volume",
            "volume": "volume",
            
            # Adj Close
            "adj close": "adj close",
            "adjusted": "adj close",
        }
        
        # Normaliser les noms de colonnes (minuscules, trim)
        df.columns = df.columns.str.strip().str.lower()
        
        # Appliquer le mapping
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        log_messages.append(f"✅ Colonnes détectées: {list(df.columns)}")
        
        # === ÉTAPE 2: Conversion du timestamp ===
        if "timestamp" not in df.columns:
            raise ValueError("❌ Aucune colonne de date/temps trouvée!")
        
        log_messages.append("📅 Conversion des dates...")
        
        # Détecter le format de date
        sample_date = str(df["timestamp"].iloc[0])
        
        if "/" in sample_date:
            # Format MM/DD/YYYY ou DD/MM/YYYY
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%Y")
            except:
                df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d/%m/%Y")
        elif "-" in sample_date:
            # Format YYYY-MM-DD ou DD-MM-YYYY
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        log_messages.append(f"✅ Dates converties ({df['timestamp'].min()} → {df['timestamp'].max()})")
        
        # === ÉTAPE 3: Nettoyage des prix (enlever virgules, convertir) ===
        log_messages.append("💰 Nettoyage des prix...")
        
        for col in ["close", "open", "high", "low"]:
            if col in df.columns:
                # Supprimer les virgules dans les nombres
                df[col] = df[col].astype(str).str.replace(",", "")
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # === ÉTAPE 4: Conversion du volume (K, M, B) ===
        if "volume" in df.columns:
            log_messages.append("📊 Conversion du volume...")
            
            def parse_volume(v):
                if pd.isna(v):
                    return 0.0
                if isinstance(v, (int, float)):
                    return float(v)
                
                v_str = str(v).strip().upper().replace(",", "")
                
                # Gérer les suffixes K, M, B
                if v_str.endswith("K"):
                    return float(v_str[:-1]) * 1_000
                elif v_str.endswith("M"):
                    return float(v_str[:-1]) * 1_000_000
                elif v_str.endswith("B"):
                    return float(v_str[:-1]) * 1_000_000_000
                else:
                    try:
                        return float(v_str)
                    except:
                        return 0.0
            
            df["volume"] = df["volume"].apply(parse_volume)
        else:
            # Si pas de volume, créer une colonne par défaut
            df["volume"] = 1000000
            log_messages.append("⚠️ Pas de volume → valeur par défaut ajoutée")
        
        # === ÉTAPE 5: Ajouter adj close si manquant ===
        if "adj close" not in df.columns:
            df["adj close"] = df["close"]
        
        # === ÉTAPE 6: Réorganiser les colonnes ===
        final_columns = ["timestamp", "open", "high", "low", "close", "adj close", "volume"]
        
        # Vérifier que toutes les colonnes essentielles existent
        missing = [c for c in ["timestamp", "open", "high", "low", "close"] if c not in df.columns]
        if missing:
            raise ValueError(f"❌ Colonnes manquantes après nettoyage: {missing}")
        
        df = df[final_columns]
        
        # === ÉTAPE 7: Supprimer les valeurs aberrantes ===
        log_messages.append("🧹 Nettoyage final...")
        
        # Supprimer les lignes avec NaN
        before_clean = len(df)
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)
        after_clean = len(df)
        
        if before_clean != after_clean:
            log_messages.append(f"⚠️ {before_clean - after_clean} lignes avec NaN supprimées")
        
        # Trier par date
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Vérifier cohérence OHLC
        df = df[
            (df["high"] >= df["low"]) &
            (df["high"] >= df["open"]) &
            (df["high"] >= df["close"]) &
            (df["low"] <= df["open"]) &
            (df["low"] <= df["close"])
        ]
        
        log_messages.append(f"✅ Données nettoyées: {len(df)} bougies valides")
        
        return df, "\n".join(log_messages)
        
    except Exception as e:
        error_msg = f"❌ Erreur lors du nettoyage: {str(e)}\n{traceback.format_exc()}"
        log_messages.append(error_msg)
        return None, "\n".join(log_messages)


# =========================================================
#  FONCTION POUR CHARGER DYNAMIQUEMENT LA STRATÉGIE
# =========================================================
def load_strategy_from_upload(uploaded_file):
    """
    Charge dynamiquement le module hyperbot_core depuis un fichier uploadé
    """
    try:
        # Lire le contenu du fichier
        content = uploaded_file.read().decode('utf-8')
        
        # Créer un fichier temporaire
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        # Charger le module dynamiquement
        spec = importlib.util.spec_from_file_location("hyperbot_core", tmp_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules['hyperbot_core'] = module
        spec.loader.exec_module(module)
        
        # Nettoyer
        os.unlink(tmp_path)
        
        return module, None
        
    except Exception as e:
        return None, str(e)


# =========================================================
#  FONCTION POUR CHARGER LE CSV (avec auto-clean)
# =========================================================
@st.cache_data
def load_csv_safely(uploaded_file) -> tuple[pd.DataFrame, str]:
    """Charge le CSV OHLCV avec nettoyage automatique"""
    try:
        # Lire le CSV brut
        content = uploaded_file.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(content))
        
        # Appliquer le nettoyage automatique
        cleaned_df, log = auto_clean_data(df)
        
        if cleaned_df is None:
            return None, log
        
        return cleaned_df, log
        
    except Exception as e:
        error_log = f"❌ Erreur lors du chargement CSV: {e}\n{traceback.format_exc()}"
        return None, error_log


# =========================================================
#  DÉTECTION AUTOMATIQUE DU TYPE D'ACTIF
# =========================================================
def detect_asset_type(df: pd.DataFrame) -> str:
    """Détecte si c'est du Crypto ou Forex"""
    if df is None or df.empty:
        return "unknown"
    
    avg_price = df['close'].mean()
    volatility = ((df['high'] - df['low']).mean() / df['close'].mean()) * 100
    
    if avg_price > 500:
        return "crypto"
    elif avg_price < 10:
        return "forex"
    else:
        return "crypto" if volatility > 2 else "forex"


# =========================================================
#  INTERFACE PRINCIPALE
# =========================================================
def main():
    # Titre
    st.title("🚀 HyperBot Trading Strategy Tester")
    st.markdown("### 📊 Testez votre stratégie avec nettoyage automatique des données")
    
    # Sidebar pour les uploads
    with st.sidebar:
        st.header("📁 Upload de Fichiers")
        
        # Upload stratégie
        st.subheader("1️⃣ Stratégie Python")
        strategy_file = st.file_uploader(
            "Upload hyperbot_core.py",
            type=["py"],
            help="Uploadez votre fichier de stratégie modifié"
        )
        
        # Upload données
        st.subheader("2️⃣ Données Historiques")
        data_file = st.file_uploader(
            "Upload CSV OHLCV",
            type=["csv"],
            help="Supporte: Investing.com, IPMS, Yahoo Finance, etc."
        )
        
        st.markdown("---")
        
        # Infos
        with st.expander("ℹ️ Formats Supportés"):
            st.markdown("""
            **Sources compatibles:**
            - 📈 Investing.com
            - 💹 IPMS
            - 📊 Yahoo Finance
            - 🌐 TradingView
            - Et autres...
            
            **Colonnes reconnues:**
            - Date/Time/Timestamp
            - Open, High, Low, Close
            - Volume (avec K, M, B)
            - Prix avec virgules (ex: "1,234.56")
            
            ✨ **Nettoyage automatique !**
            """)
    
    # Zone principale
    if not strategy_file or not data_file:
        st.info("👈 Uploadez d'abord votre stratégie et vos données dans la sidebar")
        
        # Instructions
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📝 Étape 1: Uploader la Stratégie")
            st.code("""
# Votre fichier hyperbot_core.py doit contenir:
class StrategyConfig:
    ...
    
class HyperBotOptimized:
    ...
            """, language="python")
        
        with col2:
            st.markdown("### 📊 Étape 2: Uploader les Données")
            st.markdown("**Formats acceptés:**")
            st.code("""
# Investing.com
Date,Price,Open,High,Low,Vol.,Change %
12/01/2024,1.0523,1.0510,1.0530,1.0500,41.95K,-0.12%

# Format standard
timestamp,open,high,low,close,volume
2024-01-01,42000,42500,41800,42300,1500000
            """, language="csv")
        
        return
    
    # Charger la stratégie
    with st.spinner("🔄 Chargement de la stratégie..."):
        module, error = load_strategy_from_upload(strategy_file)
        
        if error:
            st.error(f"❌ Erreur lors du chargement de la stratégie:\n```\n{error}\n```")
            with st.expander("🔍 Stack trace complet"):
                st.code(traceback.format_exc())
            return
        
        st.success("✅ Stratégie chargée avec succès!")
    
    # Charger et nettoyer les données
    with st.spinner("🧹 Chargement et nettoyage des données..."):
        df, clean_log = load_csv_safely(data_file)
        
        # Afficher le log de nettoyage
        with st.expander("📋 Log de Nettoyage des Données"):
            st.code(clean_log)
        
        if df is None:
            st.error("❌ Impossible de charger/nettoyer le fichier CSV")
            return
        
        st.success(f"✅ {len(df)} bougies chargées et nettoyées")
    
    # Afficher aperçu des données
    with st.expander("👀 Aperçu des Données Nettoyées"):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Nombre de bougies", len(df))
        col2.metric("Date début", df['timestamp'].iloc[0].strftime('%Y-%m-%d'))
        col3.metric("Date fin", df['timestamp'].iloc[-1].strftime('%Y-%m-%d'))
        col4.metric("Prix moyen", f"${df['close'].mean():.2f}")
        
        st.dataframe(df.head(10))
    
    # Détection auto du type
    asset_type = detect_asset_type(df)
    
    # Sélection du type d'actif
    st.markdown("---")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### ⚙️ Configuration")
        asset_type_selected = st.selectbox(
            "Type d'actif",
            ["Auto-détection", "Crypto", "Forex"],
            index=0
        )
        
        if asset_type_selected == "Crypto":
            asset_type = "crypto"
        elif asset_type_selected == "Forex":
            asset_type = "forex"
    
    with col2:
        if asset_type == "forex":
            st.info("💱 Mode FOREX détecté - Paramètres adaptés automatiquement")
        else:
            st.info("₿ Mode CRYPTO détecté - Paramètres standards")
    
    # Configuration des paramètres
    st.markdown("### 🎛️ Paramètres de la Stratégie")
    
    # Créer des onglets pour organiser les paramètres
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Indicateurs", "💰 Risque", "🎯 TP/SL", "⚙️ Avancé"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            if asset_type == "forex":
                ema_uf = st.number_input("EMA Ultra Fast", value=8, min_value=3, max_value=20)
                ema_f = st.number_input("EMA Fast", value=21, min_value=5, max_value=50)
                ema_t = st.number_input("EMA Trend", value=55, min_value=20, max_value=200)
            else:
                ema_uf = st.number_input("EMA Ultra Fast", value=5, min_value=3, max_value=20)
                ema_f = st.number_input("EMA Fast", value=13, min_value=5, max_value=50)
                ema_t = st.number_input("EMA Trend", value=34, min_value=20, max_value=200)
        
        with col2:
            rsi_len = st.number_input("RSI Length", value=14 if asset_type=="forex" else 9)
            atr_len = st.number_input("ATR Length", value=14 if asset_type=="forex" else 10)
        
        with col3:
            if asset_type == "forex":
                min_adx = st.number_input("Min ADX", value=15, min_value=5, max_value=30)
                rsi_min = st.number_input("RSI Min", value=25, min_value=10, max_value=40)
                rsi_max = st.number_input("RSI Max", value=75, min_value=60, max_value=90)
            else:
                min_adx = st.number_input("Min ADX", value=12, min_value=5, max_value=30)
                rsi_min = st.number_input("RSI Min", value=30, min_value=10, max_value=40)
                rsi_max = st.number_input("RSI Max", value=80, min_value=60, max_value=90)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            if asset_type == "forex":
                base_risk = st.slider("Risque de Base (%)", 1.0, 10.0, 3.0, 0.5)
            else:
                base_risk = st.slider("Risque de Base (%)", 1.0, 20.0, 10.0, 0.5)
            
            initial_capital = st.number_input("Capital Initial ($)", value=10000, step=1000)
        
        with col2:
            compound = st.checkbox("Compound Agressif", value=False if asset_type=="forex" else True)
            enable_protection = st.checkbox("Protection Sommets", value=False)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            if asset_type == "forex":
                sl_atr = st.number_input("SL ATR", value=2.0, step=0.1)
                tp_scalp = st.number_input("TP Scalp (ATR)", value=1.5, step=0.1)
                tp_swing = st.number_input("TP Swing (ATR)", value=3.0, step=0.5)
                tp_moon = st.number_input("TP Moonshot (ATR)", value=6.0, step=0.5)
            else:
                sl_atr = st.number_input("SL ATR", value=1.2, step=0.1)
                tp_scalp = st.number_input("TP Scalp (ATR)", value=1.5, step=0.1)
                tp_swing = st.number_input("TP Swing (ATR)", value=4.0, step=0.5)
                tp_moon = st.number_input("TP Moonshot (ATR)", value=12.0, step=0.5)
        
        with col2:
            scalp_percent = st.slider("Scalp Percent", 10.0, 50.0, 25.0 if asset_type=="forex" else 20.0)
            breakeven_atr = st.number_input("Breakeven ATR", value=0.8, step=0.1)
    
    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            ultra_mode = st.checkbox("Ultra Mode", value=False if asset_type=="forex" else True)
            use_trailing = st.checkbox("Use Trailing Stop", value=True)
            use_volume = st.checkbox("Volume Filter", value=False)
        
        with col2:
            commission = st.number_input("Commission (%)", value=0.002 if asset_type=="forex" else 0.1, format="%.3f")
            slippage = st.number_input("Slippage (%)", value=0.001 if asset_type=="forex" else 0.05, format="%.3f")
    
    # Bouton de lancement
    st.markdown("---")
    
    if st.button("🚀 LANCER LE BACKTEST", type="primary", use_container_width=True):
        # Créer la configuration
        try:
            config = module.StrategyConfig(
                # Indicateurs
                ema_ultra_fast=ema_uf,
                ema_fast=ema_f,
                ema_trend=ema_t,
                rsi_len=rsi_len,
                atr_len=atr_len,
                
                # Risque
                base_risk=base_risk,
                initial_capital=initial_capital,
                compound_aggressive=compound,
                
                # TP/SL
                sl_atr_base=sl_atr,
                tp_scalp=tp_scalp,
                tp_swing=tp_swing,
                tp_moonshot=tp_moon,
                scalp_percent=scalp_percent,
                breakeven_atr=breakeven_atr,
                
                # Filtres
                min_adx=min_adx,
                rsi_min=rsi_min,
                rsi_max=rsi_max,
                ultra_mode=ultra_mode,
                use_trailing=use_trailing,
                use_volume_filter=use_volume,
                
                # Protection
                enable_summit_protection=enable_protection,
                
                # Frais
                commission=commission,
                slippage=slippage,
            )
            
            # Créer la stratégie
            strategy = module.HyperBotOptimized(config)
            
            # Lancer le backtest
            with st.spinner("⏳ Backtest en cours..."):
                results_df = strategy.run_backtest(df.copy(), verbose=False)
                stats = strategy.get_statistics()
            
            # AFFICHAGE DES RÉSULTATS
            st.success("✅ Backtest terminé!")
            
            # KPIs principaux
            st.markdown("### 📊 Résultats Globaux")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            col1.metric(
                "💰 Capital Final",
                f"${stats['final_equity']:,.0f}",
                f"${stats['final_equity'] - initial_capital:,.0f}"
            )
            col2.metric(
                "📈 Rendement",
                f"{stats['total_return_pct']:.2f}%",
                delta_color="normal" if stats['total_return_pct'] > 0 else "inverse"
            )
            col3.metric(
                "🎯 Win Rate",
                f"{stats['win_rate']:.1f}%"
            )
            col4.metric(
                "📉 Max Drawdown",
                f"{stats['max_drawdown']:.2f}%",
                delta_color="inverse"
            )
            col5.metric(
                "🔄 Trades",
                f"{stats['total_trades']}"
            )
            
            # Détails supplémentaires
            st.markdown("### 📋 Détails des Performances")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Profit Factor", f"{stats['profit_factor']:.2f}")
                st.metric("Trades/Mois", f"{stats['trades_per_month']:.1f}")
            
            with col2:
                st.metric("Gain Moyen", f"${stats['avg_win']:,.2f}")
                st.metric("Perte Moyenne", f"${stats['avg_loss']:,.2f}")
            
            with col3:
                st.metric("Durée (jours)", stats['duration_days'])
                st.metric("Rendement Mensuel", f"{stats['monthly_return_pct']:.2f}%")
            
            # Graphique Equity Curve
            st.markdown("### 📈 Courbe d'Équité")
            equity_df = pd.DataFrame({
                'Equity': strategy.equity_curve
            })
            st.line_chart(equity_df)
            
            # Liste des trades
            if strategy.trades:
                st.markdown("### 🔍 Liste des Trades")
                trades_df = pd.DataFrame(strategy.trades)
                
                # Filtres
                col1, col2, col3 = st.columns(3)
                with col1:
                    filter_side = st.multiselect(
                        "Direction",
                        options=trades_df['side'].unique(),
                        default=trades_df['side'].unique()
                    )
                with col2:
                    filter_result = st.multiselect(
                        "Résultat",
                        options=['Gagnant', 'Perdant'],
                        default=['Gagnant', 'Perdant']
                    )
                
                # Appliquer filtres
                filtered_trades = trades_df[
                    (trades_df['side'].isin(filter_side)) &
                    (trades_df['pnl'] > 0 if 'Gagnant' in filter_result else True) &
                    (trades_df['pnl'] <= 0 if 'Perdant' in filter_result else True)
                ]
                
                st.dataframe(
                    filtered_trades,
                    use_container_width=True,
                    height=400
                )
                
                # Download CSV
                csv = filtered_trades.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Télécharger les trades (CSV)",
                    csv,
                    f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
            
            # Warning si pas de trades
            if stats['total_trades'] == 0:
                st.warning("""
                ⚠️ **Aucun trade détecté!**
                
                Causes possibles:
                - Filtres trop stricts (ADX, RSI, volatilité)
                - Pas de croisements d'EMAs sur la période
                - Mode ultra_mode trop restrictif pour le Forex
                
                💡 Solutions:
                - Réduire min_adx
                - Désactiver ultra_mode pour le Forex
                - Vérifier que les données contiennent des tendances
                """)
        
        except Exception as e:
            st.error(f"❌ Erreur lors du backtest:\n```\n{str(e)}\n```")
            with st.expander("🔍 Stack trace"):
                st.code(traceback.format_exc())


# =========================================================
#  FOOTER
# =========================================================
def show_footer():
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📚 Auto-Clean Data**")
        st.markdown("✨ Nettoyage automatique")
    
    with col2:
        st.markdown("**⚙️ Multi-Source**")
        st.markdown("Investing.com, IPMS, Yahoo...")
    
    with col3:
        st.markdown("**🤖 HyperBot v2.1**")
        st.markdown("Trading Strategy Tester")


# =========================================================
#  LANCEMENT
# =========================================================
if __name__ == "__main__":
    main()
    show_footer()
