"""
Live News Fetcher for Indian Financial Markets
Uses NewsAPI to fetch real-time financial news
"""
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict, Optional

class LiveNewsFetcher:
    """Fetches live financial news from NewsAPI"""
    
    def __init__(self, api_key: str = "69c8077bb5164190a1127f11c1f9ad4a"):
        """
        Initialize NewsAPI client
        
        Args:
            api_key: NewsAPI key
        """
        self.api_key = api_key
        self.client = NewsApiClient(api_key=api_key)
        
        # Indian financial news sources
        self.sources = [
            'the-times-of-india',
            'the-hindu',
            'google-news-in'
        ]
        
        # Financial keywords for filtering
        self.keywords = [
            'NIFTY', 'NIFTY 50', 'NSE', 'BSE', 'Sensex',
            'stock market', 'share market', 'equity',
            'trading', 'investor', 'stock price',
            'market index', 'Indian stocks', 'Mumbai market',
            'Dalal Street', 'financial market', 'market crash',
            'market rally', 'bull market', 'bear market',
            'IPO', 'mutual fund', 'portfolio'
        ]
    
    def fetch_latest_news(self, 
                         query: str = "NIFTY OR stock market OR BSE OR NSE",
                         hours_back: int = 24,
                         max_articles: int = 20,
                         language: str = 'en') -> List[Dict]:
        """
        Fetch latest financial news articles
        
        Args:
            query: Search query for news
            hours_back: How many hours back to search
            max_articles: Maximum number of articles to fetch
            language: Language code (default: 'en')
            
        Returns:
            List of news articles with metadata
        """
        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(hours=hours_back)
            
            # Fetch news
            response = self.client.get_everything(
                q=query,
                language=language,
                from_param=from_date.strftime('%Y-%m-%d'),
                to=to_date.strftime('%Y-%m-%d'),
                sort_by='publishedAt',
                page_size=max_articles
            )
            
            if response['status'] != 'ok':
                print(f"⚠️ API Error: {response.get('message', 'Unknown error')}")
                return []
            
            articles = response.get('articles', [])
            
            # Process articles
            processed_articles = []
            for article in articles:
                # Filter for financial relevance
                if self._is_financial_news(article):
                    processed_articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'content': article.get('content', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source', {}).get('name', 'Unknown'),
                        'published_at': article.get('publishedAt', ''),
                        'author': article.get('author', 'Unknown'),
                        'full_text': self._combine_article_text(article)
                    })
            
            print(f"✅ Fetched {len(processed_articles)} financial news articles")
            return processed_articles
            
        except Exception as e:
            print(f"❌ Error fetching news: {e}")
            return []
    
    def fetch_nifty_news(self, hours_back: int = 12, max_articles: int = 15) -> List[Dict]:
        """
        Fetch NIFTY 50 specific news
        
        Args:
            hours_back: How many hours back to search
            max_articles: Maximum number of articles
            
        Returns:
            List of NIFTY-related news articles
        """
        query = 'NIFTY 50 OR NIFTY OR NSE India'
        return self.fetch_latest_news(query, hours_back, max_articles)
    
    def fetch_top_headlines_india(self, category: str = 'business', max_articles: int = 10) -> List[Dict]:
        """
        Fetch top business headlines from India
        
        Args:
            category: News category (business, general, etc.)
            max_articles: Maximum number of articles
            
        Returns:
            List of top headline articles
        """
        try:
            response = self.client.get_top_headlines(
                country='in',
                category=category,
                page_size=max_articles
            )
            
            if response['status'] != 'ok':
                return []
            
            articles = response.get('articles', [])
            
            # Process articles
            processed_articles = []
            for article in articles:
                if self._is_financial_news(article):
                    processed_articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'content': article.get('content', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source', {}).get('name', 'Unknown'),
                        'published_at': article.get('publishedAt', ''),
                        'author': article.get('author', 'Unknown'),
                        'full_text': self._combine_article_text(article)
                    })
            
            print(f"✅ Fetched {len(processed_articles)} top headlines")
            return processed_articles
            
        except Exception as e:
            print(f"❌ Error fetching headlines: {e}")
            return []
    
    def fetch_combined_news(self, max_total: int = 20) -> List[Dict]:
        """
        Fetch combined news from multiple sources
        
        Args:
            max_total: Maximum total articles to return
            
        Returns:
            Combined list of news articles
        """
        all_articles = []
        
        # Fetch NIFTY specific news
        nifty_news = self.fetch_nifty_news(hours_back=24, max_articles=10)
        all_articles.extend(nifty_news)
        
        # Fetch top business headlines
        headlines = self.fetch_top_headlines_india(max_articles=10)
        all_articles.extend(headlines)
        
        # Remove duplicates based on title
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            title = article['title'].lower().strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_articles.append(article)
        
        # Sort by published date (most recent first)
        unique_articles.sort(
            key=lambda x: x.get('published_at', ''), 
            reverse=True
        )
        
        # Limit to max_total
        return unique_articles[:max_total]
    
    def _is_financial_news(self, article: Dict) -> bool:
        """
        Check if article is related to financial markets
        
        Args:
            article: News article dict
            
        Returns:
            True if financial news, False otherwise
        """
        text = f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}".lower()
        
        # Check for financial keywords
        for keyword in self.keywords:
            if keyword.lower() in text:
                return True
        
        return False
    
    def _combine_article_text(self, article: Dict) -> str:
        """
        Combine article title, description, and content into full text
        
        Args:
            article: News article dict
            
        Returns:
            Combined text
        """
        parts = []
        
        if article.get('title'):
            parts.append(article['title'])
        
        if article.get('description'):
            parts.append(article['description'])
        
        if article.get('content'):
            # Remove [+XXX chars] suffix that NewsAPI adds
            content = article['content']
            if '[+' in content:
                content = content.split('[+')[0].strip()
            parts.append(content)
        
        return ' '.join(parts)
    
    def get_news_summary(self, articles: List[Dict]) -> Dict:
        """
        Get summary statistics of fetched news
        
        Args:
            articles: List of news articles
            
        Returns:
            Summary statistics dict
        """
        if not articles:
            return {
                'total_articles': 0,
                'sources': [],
                'date_range': 'N/A',
                'avg_length': 0
            }
        
        sources = list(set([a['source'] for a in articles]))
        dates = [a['published_at'] for a in articles if a.get('published_at')]
        
        text_lengths = [len(a['full_text']) for a in articles if a.get('full_text')]
        avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
        
        return {
            'total_articles': len(articles),
            'sources': sources,
            'date_range': f"{min(dates) if dates else 'N/A'} to {max(dates) if dates else 'N/A'}",
            'avg_length': int(avg_length),
            'earliest': min(dates) if dates else 'N/A',
            'latest': max(dates) if dates else 'N/A'
        }


def test_news_fetcher():
    """Test the news fetcher"""
    print("\n" + "="*80)
    print("Testing Live News Fetcher")
    print("="*80 + "\n")
    
    fetcher = LiveNewsFetcher()
    
    # Test 1: Fetch NIFTY news
    print("📰 Fetching NIFTY 50 news...")
    nifty_news = fetcher.fetch_nifty_news(hours_back=24, max_articles=5)
    print(f"Found {len(nifty_news)} NIFTY articles\n")
    
    if nifty_news:
        print("Sample article:")
        article = nifty_news[0]
        print(f"  Title: {article['title'][:100]}...")
        print(f"  Source: {article['source']}")
        print(f"  Published: {article['published_at']}")
        print(f"  Text length: {len(article['full_text'])} chars\n")
    
    # Test 2: Fetch combined news
    print("📰 Fetching combined financial news...")
    combined = fetcher.fetch_combined_news(max_total=10)
    print(f"Found {len(combined)} total articles\n")
    
    # Test 3: Get summary
    summary = fetcher.get_news_summary(combined)
    print("Summary:")
    print(f"  Total articles: {summary['total_articles']}")
    print(f"  Sources: {', '.join(summary['sources'])}")
    print(f"  Average length: {summary['avg_length']} chars")
    print(f"  Date range: {summary['earliest']} to {summary['latest']}")
    
    print("\n" + "="*80)
    print("✅ Test Complete")
    print("="*80 + "\n")
    
    return combined


if __name__ == "__main__":
    test_news_fetcher()
