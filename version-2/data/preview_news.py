"""
Preview the fetched financial news data
"""

import pandas as pd
from pathlib import Path

# Find the latest news file
data_dir = Path(".")
news_files = list(data_dir.glob("indian_financial_news_*.csv"))

if news_files:
    latest_file = max(news_files, key=lambda x: x.stat().st_mtime)
    print(f"📊 Indian Financial News Data Preview")
    print("=" * 80)
    print(f"📁 File: {latest_file.name}")
    
    # Read the CSV
    df = pd.read_csv(latest_file)
    
    print(f"\n📈 Dataset Summary:")
    print(f"  • Total articles: {len(df)}")
    print(f"  • Columns: {', '.join(df.columns)}")
    print(f"  • Date range: {df['date'].min()} to {df['date'].max()}")
    
    print(f"\n📰 Top 5 Headlines:")
    for idx, row in df.head(5).iterrows():
        print(f"  {idx+1}. [{row['date']}] {row['title'][:60]}...")
        print(f"     Source: {row['source']} | Sentiment: {row['sentiment']}")
    
    print(f"\n💭 Sentiment Distribution:")
    sentiment_counts = df['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  • {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
    
    print(f"\n📺 Top News Sources:")
    for source, count in df['source'].value_counts().head(5).items():
        print(f"  • {source}: {count} articles")
    
    print(f"\n🔑 Keyword Distribution:")
    for keyword, count in df['keyword'].value_counts().items():
        print(f"  • {keyword}: {count} articles")
    
    print("\n" + "=" * 80)
    print("✅ Data preview complete!")
else:
    print("❌ No news data files found")
