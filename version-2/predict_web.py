"""
Streamlit Web Interface for NIFTY 50 Market Prediction
======================================================

Run with: streamlit run predict_web.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from predict_market import MarketPredictor
from datetime import datetime
import json
from pathlib import Path

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

# Initialize predictor
@st.cache_resource
def load_predictor(model_path):
    try:
        return MarketPredictor(model_path=model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

with st.spinner("Loading model..."):
    predictor = load_predictor(model_path)

if predictor is None:
    st.error("❌ Failed to load predictor. Please check the model path.")
    st.stop()

st.sidebar.success("✅ Model loaded successfully!")
st.sidebar.markdown(f"**Sequence Length:** {predictor.config['sequence_length']} minutes")
st.sidebar.markdown(f"**Features:** {predictor.config['n_features']}")

# Main interface
tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📊 Batch Analysis", "📈 Real-time Monitor"])

# Tab 1: Single Prediction
with tab1:
    st.header("Make a Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📰 News Input")
        
        # Get example news from session state if available
        default_news = st.session_state.get('example_news', '')
        
        news_text = st.text_area(
            "Enter news text (optional):",
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
    
    with col2:
        st.subheader("📊 Market Data")
        if Path(market_data_path).exists():
            df = pd.read_csv(market_data_path)
            st.success(f"✅ {len(df):,} data points loaded")
            st.metric("Latest Close", f"₹{df['close'].iloc[-1]:.2f}")
        else:
            st.error("❌ Market data file not found")
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔮 Make Prediction", type="primary", use_container_width=True):
        with st.spinner("Analyzing data..."):
            result = predictor.predict(
                market_csv_path=market_data_path,
                news_text=news_text if news_text.strip() else None
            )
        
        if result:
            # Display prediction
            st.markdown("## 🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if result['prediction'] == 'UP':
                    st.markdown('<p class="prediction-up">▲ UP</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="prediction-down">▼ DOWN</p>', unsafe_allow_html=True)
            
            with col2:
                st.metric(
                    "Probability",
                    f"{result['probability']:.4f}",
                    delta=f"{(result['probability']-0.5)*100:.1f}%"
                )
            
            with col3:
                st.metric(
                    "Confidence",
                    f"{result['confidence']*100:.2f}%"
                )
            
            # Signal strength
            if result['confidence'] > 0.7:
                st.success("💪 STRONG SIGNAL - High confidence prediction")
            elif result['confidence'] > 0.6:
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

# Tab 3: Real-time Monitor
with tab3:
    st.header("Real-time Market Monitor")
    
    st.info("💡 This feature will continuously monitor and predict market movements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        refresh_interval = st.slider("Refresh Interval (seconds)", 60, 600, 300)
    
    with col2:
        auto_refresh = st.checkbox("Enable Auto-refresh")
    
    if auto_refresh:
        st.warning("⏰ Auto-refresh enabled. Predictions will update every {} seconds".format(refresh_interval))
        
        # Placeholder for real-time updates
        placeholder = st.empty()
        
        import time
        
        while auto_refresh:
            with placeholder.container():
                st.markdown(f"### 🕐 Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Make prediction
                result = predictor.predict(
                    market_csv_path=market_data_path,
                    news_text=None
                )
                
                if result:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if result['prediction'] == 'UP':
                            st.success(f"▲ UP ({result['confidence']*100:.1f}% confidence)")
                        else:
                            st.error(f"▼ DOWN ({result['confidence']*100:.1f}% confidence)")
                    
                    with col2:
                        st.metric("Probability", f"{result['probability']:.4f}")
                
                st.markdown("---")
            
            time.sleep(refresh_interval)
    else:
        st.info("Enable auto-refresh to start monitoring")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>NIFTY 50 Market Predictor</strong> | Model AUC: 99.25% | Accuracy: 94.40%</p>
    <p style='color: gray; font-size: 12px'>⚠️ Disclaimer: For educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
