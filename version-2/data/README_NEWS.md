# Financial News Data - README

## 📰 Indian Financial News Dataset

This directory contains fetched financial news data for Indian markets, focusing on NIFTY 50, BSE Sensex, and Indian economy.

---

## 📁 Files Generated

### CSV Files
- `indian_financial_news_YYYYMMDD_HHMMSS.csv` - Complete news articles with metadata

### JSON Files
- `indian_financial_news_YYYYMMDD_HHMMSS.json` - Same data in JSON format
- `news_summary_YYYYMMDD_HHMMSS.json` - Summary statistics

### Scripts
- `news_getter.py` - Main script to fetch financial news
- `preview_news.py` - Preview script to display fetched data

---

## 📊 Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Publication date and time |
| `title` | string | Article headline |
| `description` | string | Brief description/summary |
| `source` | string | News source (e.g., Economic Times, Moneycontrol) |
| `url` | string | Article URL |
| `keyword` | string | Search keyword used (NIFTY 50, Indian stocks, etc.) |
| `sentiment` | string | Sentiment classification (positive, neutral, negative) |
| `relevance_score` | float | Relevance score (0-1) |

---

## 🚀 Usage

### Fetch New Data
```powershell
cd f:\DL-Project\version-2\data
F:/anaconda/envs/DL-Project/python.exe news_getter.py
```

### Preview Latest Data
```powershell
F:/anaconda/envs/DL-Project/python.exe preview_news.py
```

### Load Data in Python
```python
import pandas as pd

# Load the latest news data
df = pd.read_csv('indian_financial_news_20251026_145033.csv')

# Basic analysis
print(f"Total articles: {len(df)}")
print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
print(f"Top sources:\n{df['source'].value_counts()}")

# Filter by sentiment
positive_news = df[df['sentiment'] == 'positive']
print(f"Positive news: {len(positive_news)} articles")

# Filter by date
recent_news = df[df['date'] >= '2025-10-24']
print(f"Recent news (last 2 days): {len(recent_news)} articles")
```

---

## 📈 Latest Dataset Summary

**Fetch Date:** 2025-10-26 14:50:33  
**Total Articles:** 50  
**Date Range:** 2025-10-20 to 2025-10-25 (5 days)

### Sentiment Distribution
- ✅ **Positive:** 25 articles (50%)
- ⚪ **Neutral:** 18 articles (36%)
- ⚠️ **Negative:** 7 articles (14%)

### Top News Sources
1. CNBC-TV18 (11 articles)
2. Moneycontrol (8 articles)
3. Bloomberg India (8 articles)
4. Financial Express (6 articles)
5. Business Standard (5 articles)

### Keywords Covered
- BSE Sensex (14 articles)
- Indian economy (12 articles)
- Indian stocks (9 articles)
- NSE India (8 articles)
- NIFTY 50 (7 articles)

---

## 🔧 API Integration

The script supports the `financial-news-api` package:

```powershell
# Install the API
pip install financial-news-api

# Set API key (if required)
$env:FINANCIAL_NEWS_API_KEY="your_api_key_here"

# Run with real API
python news_getter.py
```

**Note:** Currently running in demo mode with generated sample data. Install the API package and set your API key for real data.

---

## 💡 Use Cases

1. **Sentiment Analysis** - Analyze market sentiment from news headlines
2. **Event Detection** - Identify major market events and announcements
3. **Correlation Studies** - Correlate news sentiment with market movements
4. **Trading Signals** - Generate trading signals from news data
5. **Market Research** - Track trends and patterns in financial news

---

## 🔄 Data Updates

- **Frequency:** Run `news_getter.py` daily for fresh data
- **Retention:** Keep last 30 days of data for analysis
- **Archival:** Old files can be compressed or moved to archive folder

---

## 📝 Notes

- The current dataset is **sample data** generated for demonstration
- For production use, integrate with a real financial news API
- Sentiment scores are placeholders - implement NLP sentiment analysis for accuracy
- Consider adding more data sources for comprehensive coverage

---

**Last Updated:** October 26, 2025  
**Status:** ✅ Active - Sample data available
