# app_web.py - Application Web Complète avec Upload Dynamique
"""
🌐 HYPERBOT WEB APP
Application web permettant d'uploader dynamiquement:
- Le fichier de stratégie (hyperbot_core.py)
- Les données historiques (CSV)
"""

import streamlit as st
import pandas as pd
import io
import sys
import os
import tempfile
import importlib.util
from datetime import datetime
import traceback

# Configuration de la page
st.set_page_config(
    page_title="🚀 HyperBot Web",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
#  FONCTION POUR CHARGER LE CSV
# =========================================================
@st.cache_data
def load_csv_safely(uploaded_file) -> pd.DataFrame:
    """Charge le CSV OHLCV de manière robuste"""
    try:
        content = uploaded_file.read().decode("utf-8")
        lines = content.splitlines()

        # Trouver la ligne d'en-tête
        skip = 0
        for i, line in enumerate(lines[:10]):
            if "close" in line.lower() or "open" in line.lower():
                skip = i
                break

        df = pd.read_csv(io.StringIO(content), skiprows=skip)
        df.columns = df.columns.str.strip().str.lower()

        # Vérifier colonnes essentielles
        required = {"open", "high", "low", "close"}
        if not required.issubset(df.columns):
            raise ValueError(f"Colonnes manquantes: {required - set(df.columns)}")

        # Convertir en numérique
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Gérer les dates
        if "date" in df.columns:
            df.rename(columns={"date": "timestamp"}, inplace=True)
        
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        else:
            raise ValueError("Aucune colonne 'timestamp' ou 'date' trouvée")

        df.dropna(inplace=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
        
    except Exception as e:
        st.error(f"Erreur lors du chargement CSV: {e}")
        return None


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
    st.markdown("### 📊 Testez votre stratégie en uploadant vos fichiers")
    
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
            help="Données avec colonnes: timestamp, open, high, low, close, volume"
        )
        
        st.markdown("---")
        
        # Infos
        with st.expander("ℹ️ Informations"):
            st.markdown("""
            **Format CSV requis:**
            - timestamp (ou date)
            - open, high, low, close
            - volume (optionnel)
            
            **Stratégie:**
            - Doit contenir `HyperBotOptimized` et `StrategyConfig`
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
            st.code("""
timestamp,open,high,low,close,volume
2024-01-01,42000,42500,41800,42300,1500000
2024-01-02,42300,43000,42100,42800,1800000
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
    
    # Charger les données
    with st.spinner("📊 Chargement des données..."):
        df = load_csv_safely(data_file)
        
        if df is None:
            st.error("❌ Impossible de charger le fichier CSV")
            return
        
        st.success(f"✅ {len(df)} bougies chargées")
    
    # Afficher aperçu des données
    with st.expander("👀 Aperçu des Données"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Nombre de bougies", len(df))
        col2.metric("Date début", df['timestamp'].iloc[0].strftime('%Y-%m-%d'))
        col3.metric("Date fin", df['timestamp'].iloc[-1].strftime('%Y-%m-%d'))
        
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
        st.markdown("**📚 Documentation**")
        st.markdown("[Guide d'utilisation](#)")
    
    with col2:
        st.markdown("**⚙️ Configuration**")
        st.markdown("Version: 2.0")
    
    with col3:
        st.markdown("**🤖 HyperBot**")
        st.markdown("Trading Strategy Tester")


# =========================================================
#  LANCEMENT
# =========================================================
if __name__ == "__main__":
    main()
    show_footer()
