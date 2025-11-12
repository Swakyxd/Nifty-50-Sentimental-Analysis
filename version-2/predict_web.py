"""
Streamlit Web Interface for NIFTY 50 Market Prediction
======================================================

Features:
- Live NIFTY 50 price fetching (yfinance)
- Live news integration (NewsAPI)
- Real-time monitoring with auto-refresh
- Price estimation with volatility
- Batch analysis support

Run with: streamlit run predict_web.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from predict_market import MarketPredictor
from datetime import datetime
import json
from pathlib import Path
import time

# Page configuration
st.set_page_config(
    page_title="NIFTY 50 Market Predictor",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-weight: bold;
}
.prediction-up {
    color: #00ff00;
    font-size: 40px;
    font-weight: bold;
}
.prediction-down {
    color: #ff0000;
    font-size: 40px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("📈 NIFTY 50 Market Predictor")
st.markdown("---")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

model_path = st.sidebar.text_input(
    "Model Path",
    value="models/trained_local"
)

market_data_path = st.sidebar.text_input(
    "Market Data CSV",
    value="processed/nifty50_NIFTY 50_minute_featured.csv"
)

# NewsAPI Configuration
st.sidebar.markdown("---")
st.sidebar.subheader("📰 Live News (Optional)")

use_live_news = st.sidebar.checkbox("Enable Live News Fetching", value=False)
news_api_key = None

if use_live_news:
    news_api_key = st.sidebar.text_input(
        "NewsAPI Key",
        value="69c8077bb5164190a1127f11c1f9ad4a",
        type="password",
        help="Get free API key from newsapi.org (100 requests/day)"
    )
    st.sidebar.info("📡 Live news will be fetched from NewsAPI for predictions")

# Initialize predictor
@st.cache_resource
def load_predictor(model_path, api_key=None):
    try:
        return MarketPredictor(model_path=model_path, news_api_key=api_key)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to get live news (NOT cached to ensure fresh news)
def fetch_fresh_news(predictor, keywords=None, max_articles=3):
    """Fetch fresh news without caching"""
    if predictor.news_api:
        return predictor.get_live_news(keywords=keywords, max_articles=max_articles)
    return None

with st.spinner("Loading model..."):
    predictor = load_predictor(model_path, news_api_key)

if predictor is None:
    st.error("❌ Failed to load predictor. Please check the model path.")
    st.stop()

st.sidebar.success("✅ Model loaded successfully!")
st.sidebar.markdown(f"**Validation AUC:** 99.7%")
st.sidebar.markdown(f"**Sequence Length:** {predictor.config['sequence_length']} minutes")
st.sidebar.markdown(f"**Features:** {predictor.config['n_features']}")

if predictor.news_api:
    st.sidebar.success("✅ NewsAPI connected")
else:
    st.sidebar.info("ℹ️ NewsAPI not configured")

# Main interface
tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📊 Batch Analysis", " Live News Explorer"])

# Tab 1: Single Prediction
with tab1:
    st.header("Make a Prediction")
    
    # Auto-fetch live price on first load
    if 'live_price_fetched' not in st.session_state:
        with st.spinner("Fetching live NIFTY 50 price..."):
            live_price = predictor.get_live_nifty_price()
            if live_price:
                st.session_state['last_live_price'] = live_price
                st.session_state['live_price_fetched'] = True
    
    # Live price display
    col_price1, col_price2 = st.columns(2)
    
    with col_price1:
        if 'last_live_price' in st.session_state:
            st.metric("💰 Live NIFTY 50 Price", f"₹{st.session_state['last_live_price']:,.2f}", 
                     delta="Live Market", delta_color="off")
        else:
            st.info("📊 Live price not fetched yet")
    
    with col_price2:
        if st.button("🔄 Refresh Live Price", key="live_price_single"):
            with st.spinner("Fetching live price..."):
                live_price = predictor.get_live_nifty_price()
                if live_price:
                    st.session_state['last_live_price'] = live_price
                    st.success(f"✅ Updated: ₹{live_price:,.2f}")
                    st.rerun()
                else:
                    st.warning("⚠️ Could not fetch live price. Using CSV data.")
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📰 News Input")
        
        # News input options
        news_mode = st.radio(
            "News Source:",
            ["Manual Entry", "Fetch Live News", "No News (Market Only)"],
            horizontal=True
        )
        
        news_text = None
        
        if news_mode == "Manual Entry":
            # Get example news from session state if available
            default_news = st.session_state.get('example_news', '')
            
            news_text = st.text_area(
                "Enter news text:",
                value=default_news,
                height=150,
                placeholder="E.g., Positive market sentiment drives NIFTY 50 higher with strong institutional buying..."
            )
            
            # Example news
            if st.button("Load Example News"):
                st.session_state['example_news'] = """
                NIFTY 50 shows strong positive momentum as major tech companies report 
                record earnings. Market sentiment remains bullish with increased investor 
                confidence. Banking sector leads the rally with significant gains.
                """
                st.rerun()
        
        elif news_mode == "Fetch Live News":
            if predictor.news_api:
                col_btn1, col_btn2 = st.columns([1, 1])
                
                with col_btn1:
                    if st.button("📡 Fetch Latest News", key="fetch_news_single"):
                        with st.spinner("Fetching live news from NewsAPI..."):
                            live_news = fetch_fresh_news(predictor, max_articles=3)
                            if live_news:
                                st.session_state['fetched_news'] = live_news
                                st.session_state['news_fetch_time'] = datetime.now().strftime('%H:%M:%S')
                                st.session_state['news_fetch_timestamp'] = time.time()  # Add timestamp
                                st.success(f"✅ Fetched live news successfully!")
                                st.rerun()
                            else:
                                st.warning("⚠️ No news articles found")
                
                with col_btn2:
                    if st.button("🗑️ Clear News", key="clear_news_single"):
                        if 'fetched_news' in st.session_state:
                            del st.session_state['fetched_news']
                        if 'news_fetch_time' in st.session_state:
                            del st.session_state['news_fetch_time']
                        st.rerun()
                
                if 'fetched_news' in st.session_state:
                    # Show fetch time
                    if 'news_fetch_time' in st.session_state:
                        st.caption(f"🕐 Fetched at: {st.session_state['news_fetch_time']}")
                    
                    # Display news with character limit
                    news_preview = st.session_state['fetched_news'][:800] + "..." if len(st.session_state['fetched_news']) > 800 else st.session_state['fetched_news']
                    
                    st.text_area(
                        "Fetched News:",
                        value=news_preview,
                        height=150,
                        disabled=True,
                        help="Fresh news articles from the last 24 hours"
                    )
                    news_text = st.session_state['fetched_news']
                else:
                    st.info("👆 Click 'Fetch Latest News' to get current market news")
            else:
                st.warning("⚠️ NewsAPI not configured. Enable in sidebar.")
    
    with col2:
        st.subheader("📊 Market Data")
        if Path(market_data_path).exists():
            df = pd.read_csv(market_data_path)
            st.success(f"✅ {len(df):,} data points loaded")
            
            csv_price = df['close'].iloc[-1]
            st.metric("CSV Latest Close", f"₹{csv_price:.2f}", delta="Historical")
            
            # Show comparison if live price available
            if 'last_live_price' in st.session_state:
                live_price = st.session_state['last_live_price']
                price_diff = live_price - csv_price
                price_diff_pct = (price_diff / csv_price) * 100
                st.metric("Price Change", f"₹{price_diff:+,.2f}", delta=f"{price_diff_pct:+.2f}%")
        else:
            st.error("❌ Market data file not found")
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔮 Make Prediction", type="primary", use_container_width=True):
        with st.spinner("Analyzing data..."):
            result = predictor.predict(
                market_csv_path=market_data_path,
                news_text=news_text if news_text and news_text.strip() else None
            )
        
        if result:
            # Display prediction
            st.markdown("## 🎯 Prediction Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if result['prediction'] == 'UP':
                    st.markdown('<p class="prediction-up">▲ UP</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="prediction-down">▼ DOWN</p>', unsafe_allow_html=True)
            
            with col2:
                st.metric(
                    "Confidence",
                    f"{result['confidence']*100:.2f}%"
                )
            
            with col3:
                if 'current_price' in result:
                    price_source = result.get('price_source', 'Unknown')
                    st.metric(
                        f"Current Price ({price_source})",
                        f"₹{result['current_price']:,.2f}"
                    )
            
            with col4:
                if 'predicted_price' in result:
                    change_pct = result.get('price_change_percent', 0)
                    st.metric(
                        "Predicted Price",
                        f"₹{result['predicted_price']:,.2f}",
                        delta=f"{change_pct:+.4f}%"
                    )
            
            # Price estimation details
            if 'predicted_price' in result and 'current_price' in result:
                st.markdown("### 💰 Price Estimation")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    change_value = result.get('price_change_value', 0)
                    st.metric("Expected Movement", f"₹{change_value:+.2f}")
                
                with col2:
                    change_pct = result.get('price_change_percent', 0)
                    st.metric("Expected Change %", f"{change_pct:+.4f}%")
                
                with col3:
                    method = result.get('estimation_method', 'N/A')
                    st.info(f"📊 Method: {method}")
            
            # Signal strength
            if result['confidence'] > 0.8:
                st.success("💪 STRONG SIGNAL - High confidence prediction")
            elif result['confidence'] > 0.65:
                st.warning("✓ MODERATE SIGNAL - Monitor closely")
            else:
                st.error("⚠️ WEAK SIGNAL - Low confidence")
            
            # News features
            st.markdown("### 📊 News Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Companies Mentioned", result['news_features']['news_num_companies'])
            with col2:
                st.metric("Keywords Found", result['news_features']['news_num_keywords'])
            with col3:
                st.metric("Text Length", result['news_features']['news_text_length'])
            
            # Additional details in expander
            with st.expander("📋 Full Prediction Details"):
                st.json(result)
            
            # Save result
            predictions_dir = Path('predictions')
            predictions_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = predictions_dir / f'prediction_{timestamp}.json'
            
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            st.success(f"💾 Prediction saved to: {result_file}")

# Tab 2: Batch Analysis
with tab2:
    st.header("Batch News Analysis")
    
    st.markdown("Enter multiple news articles (one per line) for batch analysis:")
    
    news_batch = st.text_area(
        "News Articles:",
        height=300,
        placeholder="Article 1...\n\nArticle 2...\n\nArticle 3..."
    )
    
    if st.button("📊 Analyze Batch", type="primary"):
        if news_batch.strip():
            # Split by double newlines
            articles = [a.strip() for a in news_batch.split('\n\n') if a.strip()]
            
            with st.spinner(f"Analyzing {len(articles)} articles..."):
                results = predictor.predict_batch(
                    market_csv_path=market_data_path,
                    news_list=articles
                )
            
            # Display results
            st.markdown("## 📈 Batch Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            up_count = sum(1 for r in results if r and r['prediction'] == 'UP')
            down_count = len(results) - up_count
            avg_confidence = np.mean([r['confidence'] for r in results if r])
            
            with col1:
                st.metric("Total Articles", len(results))
            with col2:
                st.metric("UP Predictions", up_count)
            with col3:
                st.metric("DOWN Predictions", down_count)
            with col4:
                st.metric("Avg Confidence", f"{avg_confidence*100:.1f}%")
            
            # Detailed results table
            st.markdown("### Detailed Results")
            
            results_data = []
            for i, result in enumerate(results, 1):
                if result:
                    results_data.append({
                        'Article': i,
                        'Prediction': result['prediction'],
                        'Probability': f"{result['probability']:.4f}",
                        'Confidence': f"{result['confidence']*100:.2f}%",
                        'Signal': '💪 Strong' if result['confidence'] > 0.7 else '✓ Moderate' if result['confidence'] > 0.6 else '⚠️ Weak'
                    })
            
            df_results = pd.DataFrame(results_data)
            st.dataframe(df_results, use_container_width=True)
        else:
            st.warning("Please enter news articles to analyze")

# Tab 3: Live News Explorer
with tab3:
    st.header("📰 Live News Explorer")
    
    if not predictor.news_api:
        st.warning("⚠️ NewsAPI not configured. Please enable in the sidebar.")
        st.info("Get your free API key from [newsapi.org](https://newsapi.org/) (100 requests/day)")
    else:
        st.success("✅ NewsAPI is connected and ready")
        
        col1, col2 = st.columns(2)
        
        with col1:
            news_keywords = st.multiselect(
                "Select Keywords:",
                [
                    "NIFTY 50", "NSE", "Sensex", "BSE", "Indian stock market",
                    "Reliance Industries", "TCS", "HDFC Bank", "Infosys", 
                    "ICICI Bank", "Bharti Airtel", "SBI", "ITC",
                    "Bajaj Finance", "Kotak Bank", "HUL", "Axis Bank",
                    "Larsen Toubro", "Asian Paints", "Maruti Suzuki",
                    "Titan", "Wipro", "UltraTech Cement", "Adani",
                    "Indian economy", "RBI", "inflation India", "rupee"
                ],
                default=["NIFTY 50", "Reliance Industries", "TCS"]
            )
        
        with col2:
            max_articles = st.slider("Number of Articles", 1, 10, 5)
        
        if st.button("📡 Fetch News", type="primary"):
            with st.spinner("Fetching news from NewsAPI..."):
                news_text_explore = fetch_fresh_news(
                    predictor,
                    keywords=news_keywords,
                    max_articles=max_articles
                )
            
            if news_text_explore:
                st.success(f"✅ Successfully fetched news articles!")
                
                # Split into individual articles
                articles = news_text_explore.split('\n\n')
                
                st.markdown(f"### 📑 {len(articles)} Articles Found")
                
                for i, article in enumerate(articles, 1):
                    with st.expander(f"📄 Article {i}"):
                        st.write(article)
                        
                        # Extract features from this article
                        features = predictor.extract_news_features(article)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Companies", features['news_num_companies'])
                        with col2:
                            st.metric("Keywords", features['news_num_keywords'])
                        with col3:
                            st.metric("Length", features['news_text_length'])
                        
                        # Quick prediction button for this article
                        if st.button(f"🔮 Predict with Article {i}", key=f"predict_{i}"):
                            with st.spinner("Making prediction..."):
                                result = predictor.predict(
                                    market_csv_path=market_data_path,
                                    news_text=article
                                )
                            
                            if result:
                                if result['prediction'] == 'UP':
                                    st.success(f"▲ UP - Confidence: {result['confidence']*100:.2f}%")
                                else:
                                    st.error(f"▼ DOWN - Confidence: {result['confidence']*100:.2f}%")
                                
                                if 'predicted_price' in result:
                                    st.info(f"Expected change: {result['price_change_percent']:+.4f}%")
                
                # Option to use all combined news
                st.markdown("---")
                if st.button("🔮 Predict with All Articles Combined", type="primary"):
                    with st.spinner("Making prediction with combined news..."):
                        result = predictor.predict(
                            market_csv_path=market_data_path,
                            news_text=news_text_explore
                        )
                    
                    if result:
                        st.markdown("### 🎯 Combined News Prediction")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            if result['prediction'] == 'UP':
                                st.success(f"▲ UP")
                            else:
                                st.error(f"▼ DOWN")
                        
                        with col2:
                            st.metric("Confidence", f"{result['confidence']*100:.2f}%")
                        
                        with col3:
                            if 'predicted_price' in result:
                                st.metric("Predicted", f"₹{result['predicted_price']:,.2f}")
                        
                        with col4:
                            if 'price_change_percent' in result:
                                st.metric("Change", f"{result['price_change_percent']:+.4f}%")
            else:
                st.error("❌ No news articles found. Try different keywords or check API limits.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>NIFTY 50 Market Predictor</strong> | Model Validation AUC: 99.7% | Sentiment Accuracy: 99.9%</p>
    <p style='color: gray; font-size: 12px'>⚠️ Disclaimer: For educational purposes only. Not financial advice.</p>
    <p style='color: gray; font-size: 10px'>Features: Live Price (yfinance) | Live News (NewsAPI) | Volatility-based Estimation</p>
</div>
""", unsafe_allow_html=True)
