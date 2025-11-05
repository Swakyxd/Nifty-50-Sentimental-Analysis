"""
Example: How to Use the Market Predictor
=========================================

This script demonstrates different ways to use the market predictor.
"""

from predict_market import MarketPredictor
from pathlib import Path

# Example 1: Simple Prediction with News
print("=" * 80)
print("EXAMPLE 1: Basic Prediction with News")
print("=" * 80)

# Initialize predictor
predictor = MarketPredictor(model_path='models/trained_local')

# Your news text
news_text = """
NIFTY 50 shows strong positive momentum as major tech companies report 
record earnings. Market sentiment remains bullish with increased investor 
confidence. Banking sector leads the rally with significant gains.
"""

# Make prediction
result = predictor.predict(
    market_csv_path='processed/nifty50_NIFTY 50_minute_featured.csv',
    news_text=news_text
)

print("\n" + "=" * 80)
print("EXAMPLE 2: Prediction without News (Market Data Only)")
print("=" * 80)

# Prediction based only on market data (no news)
result = predictor.predict(
    market_csv_path='processed/nifty50_NIFTY 50_minute_featured.csv',
    news_text=None  # No news
)

print("\n" + "=" * 80)
print("EXAMPLE 3: Batch Predictions with Multiple News Articles")
print("=" * 80)

# Multiple news articles
news_articles = [
    "Positive economic indicators boost market sentiment across all sectors.",
    "Market experiences volatility as global concerns increase uncertainty.",
    "Strong corporate earnings drive NIFTY 50 to new highs today."
]

# Batch prediction
results = predictor.predict_batch(
    market_csv_path='processed/nifty50_NIFTY 50_minute_featured.csv',
    news_list=news_articles
)

print("\n📊 Batch Results Summary:")
for i, result in enumerate(results, 1):
    if result:
        print(f"   Article {i}: {result['prediction']} (Confidence: {result['confidence']*100:.1f}%)")

print("\n" + "=" * 80)
print("EXAMPLE 4: Real-time News Analysis")
print("=" * 80)

# Simulate real-time news from Economic Times
import requests
from datetime import datetime

def fetch_latest_news():
    """Fetch latest news (placeholder - replace with actual news API)"""
    # This is a placeholder - replace with actual news API
    return "Market shows positive signs with strong institutional buying."

# Get latest news
latest_news = fetch_latest_news()

# Make prediction
result = predictor.predict(
    market_csv_path='processed/nifty50_NIFTY 50_minute_featured.csv',
    news_text=latest_news
)

# Trading signal
if result:
    print(f"\n🎯 Trading Signal Generated:")
    print(f"   Time: {result['timestamp']}")
    print(f"   Signal: {result['prediction']}")
    print(f"   Confidence: {result['confidence']*100:.2f}%")
    
    if result['confidence'] > 0.7:
        print(f"   ✅ HIGH CONFIDENCE - Consider taking action")
    elif result['confidence'] > 0.6:
        print(f"   ⚠️ MODERATE CONFIDENCE - Monitor closely")
    else:
        print(f"   ❌ LOW CONFIDENCE - Wait for better signal")

print("\n" + "=" * 80)
print("✅ All examples completed!")
print("=" * 80)
