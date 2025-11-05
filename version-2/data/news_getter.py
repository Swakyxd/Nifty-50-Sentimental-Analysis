"""
Financial News Fetcher for Indian Markets
==========================================

Fetches recent financial news data for Indian markets using the financial-news-api.
Focuses on NIFTY 50, Indian stock market, and economic news.

Requirements:
    pip install financial-news-api

Usage:
    python news_getter.py
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Try importing the financial news api
try:
    from financial_news_api import FinancialNewsAPI
    API_AVAILABLE = True
except ImportError:
    print("⚠️ financial-news-api not installed. Using mock data instead.")
    print("   Install with: pip install financial-news-api")
    API_AVAILABLE = False


class IndianFinancialNewsGetter:
    """
    Fetches financial news related to Indian markets
    """
    
    def __init__(self, api_key=None, output_dir="data"):
        """
        Initialize the news getter
        
        Args:
            api_key (str): API key for financial news service (if required)
            output_dir (str): Directory to save news data
        """
        self.api_key = api_key or os.getenv("FINANCIAL_NEWS_API_KEY")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if API_AVAILABLE and self.api_key:
            self.api = FinancialNewsAPI(api_key=self.api_key)
        else:
            self.api = None
            print("📝 Running in demo mode - will generate sample data")
    
    def fetch_indian_market_news(self, days=5, max_articles=100):
        """
        Fetch financial news for Indian markets
        
        Args:
            days (int): Number of days to look back
            max_articles (int): Maximum number of articles to fetch
            
        Returns:
            pd.DataFrame: DataFrame with news articles
        """
        print(f"📰 Fetching Indian financial news for the last {days} days...")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        if self.api and API_AVAILABLE:
            # Use real API if available
            articles = self._fetch_real_news(start_date, end_date, max_articles)
        else:
            # Generate sample data
            articles = self._generate_sample_news(days, max_articles)
        
        # Convert to DataFrame
        df = pd.DataFrame(articles)
        
        if len(df) > 0:
            print(f"✅ Fetched {len(df)} articles")
            self._save_news_data(df)
        else:
            print("⚠️ No articles found")
        
        return df
    
    def _fetch_real_news(self, start_date, end_date, max_articles):
        """
        Fetch real news from API
        """
        articles = []
        
        # Keywords for Indian market news
        keywords = [
            "NIFTY 50",
            "Indian stock market",
            "BSE Sensex",
            "NSE India",
            "Indian economy",
            "RBI",
            "India GDP",
            "Indian stocks"
        ]
        
        try:
            for keyword in keywords:
                print(f"  Searching for: {keyword}")
                results = self.api.get_news(
                    query=keyword,
                    from_date=start_date.strftime('%Y-%m-%d'),
                    to_date=end_date.strftime('%Y-%m-%d'),
                    language='en',
                    max_results=max_articles // len(keywords)
                )
                
                for article in results:
                    articles.append({
                        'date': article.get('publishedAt', ''),
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'url': article.get('url', ''),
                        'keyword': keyword,
                        'sentiment': 'neutral'  # Placeholder
                    })
                
                if len(articles) >= max_articles:
                    break
        
        except Exception as e:
            print(f"❌ Error fetching news: {e}")
            print("   Generating sample data instead...")
            return self._generate_sample_news(
                (end_date - start_date).days, 
                max_articles
            )
        
        return articles[:max_articles]
    
    def _generate_sample_news(self, days, max_articles):
        """
        Generate sample news data for demonstration
        """
        print("📝 Generating sample Indian financial news data...")
        
        sample_titles = [
            "NIFTY 50 reaches new all-time high as investors show confidence",
            "RBI maintains repo rate at 6.5% in latest monetary policy review",
            "Indian IT sector shows strong growth in Q3 earnings",
            "FIIs inject Rs 5,000 crore into Indian markets this week",
            "Adani Group stocks surge after debt reduction announcement",
            "Reliance Industries reports record quarterly profit",
            "Indian rupee strengthens against dollar amid positive market sentiment",
            "BSE Sensex crosses 75,000 mark for the first time",
            "Tata Motors announces expansion plans for EV segment",
            "Indian banking sector shows resilience with improved asset quality",
            "HDFC Bank reports strong deposit growth in latest quarter",
            "Infosys secures major deal with European banking giant",
            "Indian pharmaceutical exports reach record high",
            "Government announces tax benefits for first-time investors",
            "Foreign exchange reserves touch new high of $650 billion",
            "Manufacturing PMI shows robust expansion in Indian economy",
            "Indian startups raise $2 billion in funding this quarter",
            "Gold prices decline as investors shift to equity markets",
            "Oil prices impact Indian inflation outlook says RBI",
            "Indian market outperforms emerging markets in returns",
        ]
        
        sample_sources = [
            "Economic Times",
            "Business Standard",
            "Moneycontrol",
            "Bloomberg India",
            "Reuters India",
            "Mint",
            "Financial Express",
            "CNBC-TV18"
        ]
        
        sample_descriptions = [
            "Market analysis shows positive trends across sectors with strong institutional support.",
            "Economic indicators suggest sustained growth momentum in the Indian economy.",
            "Experts predict continued bullish sentiment driven by domestic factors.",
            "Technical analysis indicates potential for further upside in near term.",
            "Policy measures expected to boost investor confidence and market stability.",
        ]
        
        articles = []
        end_date = datetime.now()
        
        import random
        for i in range(min(max_articles, len(sample_titles) * 3)):
            # Random date within the last 'days' days
            days_ago = random.randint(0, days)
            article_date = end_date - timedelta(
                days=days_ago,
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            # Random sentiment
            sentiment = random.choice(['positive', 'neutral', 'negative'])
            sentiment_weights = {'positive': 0.5, 'neutral': 0.3, 'negative': 0.2}
            sentiment = random.choices(
                list(sentiment_weights.keys()),
                weights=list(sentiment_weights.values())
            )[0]
            
            articles.append({
                'date': article_date.strftime('%Y-%m-%d %H:%M:%S'),
                'title': random.choice(sample_titles),
                'description': random.choice(sample_descriptions),
                'source': random.choice(sample_sources),
                'url': f"https://example.com/news/article-{i+1}",
                'keyword': random.choice([
                    "NIFTY 50", "Indian stocks", "BSE Sensex", 
                    "NSE India", "Indian economy"
                ]),
                'sentiment': sentiment,
                'relevance_score': round(random.uniform(0.6, 1.0), 2)
            })
        
        return articles
    
    def _save_news_data(self, df):
        """
        Save news data to CSV and JSON formats
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save as CSV
        csv_file = self.output_dir / f"indian_financial_news_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"💾 Saved CSV: {csv_file}")
        
        # Save as JSON
        json_file = self.output_dir / f"indian_financial_news_{timestamp}.json"
        df.to_json(json_file, orient='records', indent=2)
        print(f"💾 Saved JSON: {json_file}")
        
        # Save summary
        self._save_summary(df, timestamp)
    
    def _save_summary(self, df, timestamp):
        """
        Save a summary of the fetched news
        """
        summary = {
            'fetch_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_articles': len(df),
            'date_range': {
                'start': df['date'].min() if len(df) > 0 else 'N/A',
                'end': df['date'].max() if len(df) > 0 else 'N/A'
            },
            'sources': df['source'].value_counts().to_dict() if 'source' in df.columns else {},
            'keywords': df['keyword'].value_counts().to_dict() if 'keyword' in df.columns else {},
            'sentiment_distribution': df['sentiment'].value_counts().to_dict() if 'sentiment' in df.columns else {}
        }
        
        summary_file = self.output_dir / f"news_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"📊 Saved summary: {summary_file}")
    
    def display_news_preview(self, df, n=5):
        """
        Display a preview of the fetched news
        """
        print(f"\n📰 Preview of Latest {n} Articles:")
        print("=" * 80)
        
        for idx, row in df.head(n).iterrows():
            print(f"\n[{idx+1}] {row['date']}")
            print(f"📌 {row['title']}")
            print(f"📰 Source: {row['source']}")
            if 'sentiment' in row:
                sentiment_emoji = {'positive': '😊', 'neutral': '😐', 'negative': '😟'}
                print(f"💭 Sentiment: {sentiment_emoji.get(row['sentiment'], '😐')} {row['sentiment']}")
            print(f"🔗 {row.get('url', 'N/A')}")
            print("-" * 80)


def main():
    """
    Main function to run the news getter
    """
    print("=" * 80)
    print("Indian Financial News Fetcher")
    print("=" * 80)
    
    # Initialize the news getter
    getter = IndianFinancialNewsGetter(output_dir="../data")
    
    # Fetch news for the last 5 days
    df = getter.fetch_indian_market_news(days=5, max_articles=50)
    
    # Display preview
    if len(df) > 0:
        getter.display_news_preview(df, n=5)
        
        # Display statistics
        print(f"\n📊 Statistics:")
        print(f"  • Total articles: {len(df)}")
        print(f"  • Date range: {df['date'].min()} to {df['date'].max()}")
        
        if 'source' in df.columns:
            print(f"\n  📰 Top Sources:")
            for source, count in df['source'].value_counts().head(5).items():
                print(f"    • {source}: {count} articles")
        
        if 'sentiment' in df.columns:
            print(f"\n  💭 Sentiment Distribution:")
            for sentiment, count in df['sentiment'].value_counts().items():
                print(f"    • {sentiment.capitalize()}: {count} articles ({count/len(df)*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("✅ News fetching complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
