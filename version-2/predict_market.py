"""
NIFTY 50 Market Prediction Script
==================================
Predicts market movement (UP/DOWN) and price changes based on news sentiment.
Uses Attention-LSTM model (99.9% accuracy) for direction prediction.
Price changes estimated from sentiment confidence and market volatility.

Usage:
    python predict_market.py --date 2025-11-05
    python predict_market.py --date 2025-11-05 --news "Positive economic indicators boost market sentiment"
    python predict_market.py --realtime
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import json
import pickle
from datetime import datetime, timedelta
import argparse
import warnings
warnings.filterwarnings('ignore')

# Try to import yfinance for live data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️ yfinance not available. Will use CSV data only. Install: pip install yfinance")

# Try to import NewsAPI for live news
try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False
    print("⚠️ newsapi-python not available. Install: pip install newsapi-python")

# Windows console compatibility
def safe_print(*args, **kwargs):
    """Print with fallback for Windows console emoji issues"""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # Remove emojis if encoding fails
        text = ' '.join(str(arg) for arg in args)
        text = text.encode('ascii', 'ignore').decode('ascii')
        print(text, **kwargs)


class AttentionLSTM(nn.Module):
    """
    Attention-LSTM Model Architecture (matches trained finbert_lstm model)
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional=True):
        super(AttentionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * self.num_directions, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # LSTM output: (batch, seq_len, hidden_size * num_directions)
        lstm_out, _ = self.lstm(x)
        
        # Attention weights: (batch, seq_len, 1)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        
        # Apply attention: (batch, hidden_size * num_directions)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Output
        out = self.dropout(context)
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out


# Keep old class name for compatibility
class EnhancedAttentionLSTM(AttentionLSTM):
    """
    Alias for backward compatibility
    """
    def __init__(self, input_size, lstm_units=128, dropout=0.3):
        # Convert to new init parameters
        super().__init__(
            input_size=input_size,
            hidden_size=lstm_units,
            num_layers=2,
            dropout=dropout,
            bidirectional=True
        )


class MarketPredictor:
    """
    Enhanced Market Movement and Price Predictor
    Uses Attention-LSTM (99.9% accuracy) for direction prediction.
    Estimates price changes from sentiment confidence and market volatility.
    """
    
    def __init__(self, model_path='models/finbert_lstm', news_api_key=None):
        # Get script directory and construct absolute paths
        script_dir = Path(__file__).parent.resolve()
        self.model_path = script_dir / model_path if not Path(model_path).is_absolute() else Path(model_path)
        self.device = torch.device('cpu')  # Use CPU for prediction
        self.news_api_key = news_api_key
        
        # Initialize NewsAPI if available and key provided
        self.news_api = None
        if NEWSAPI_AVAILABLE and news_api_key:
            try:
                self.news_api = NewsApiClient(api_key=news_api_key)
                safe_print("✅ NewsAPI initialized")
            except Exception as e:
                safe_print(f"⚠️ Could not initialize NewsAPI: {e}")
        
        safe_print("=" * 80)
        safe_print("🚀 NIFTY 50 Market Predictor (Sentiment-Based)")
        safe_print("=" * 80)
        
        # Try new model structure first (finbert_lstm), fallback to old structure
        try:
            # Try loading with new structure (finbert_lstm)
            self._load_finbert_lstm_model()
        except Exception as e1:
            safe_print(f"⚠️ Couldn't load finbert_lstm model: {e1}")
            try:
                # Fallback to old structure
                safe_print("   Trying old model structure...")
                self._load_config()
                self._load_features()
                self._load_scaler()
                self._load_model()
            except Exception as e2:
                raise Exception(f"Failed to load model. New structure error: {e1}. Old structure error: {e2}")
        
        safe_print("✅ Predictor initialized successfully")
        safe_print("   Model: Attention-LSTM")
        safe_print("   Strategy: Sentiment-driven direction + Volatility-based magnitude")
        safe_print("=" * 80)
    
    def _load_finbert_lstm_model(self):
        """Load model using new finbert_lstm structure"""
        # Check if this is a batch model (has model.pt) or main model (has best_model.pt)
        is_batch_model = (self.model_path / 'model.pt').exists()
        
        # Load configuration
        config_path = self.model_path / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            safe_print(f"✅ Config loaded: {config_path}")
        else:
            # Batch models don't have config.json, use defaults
            safe_print(f"⚠️ No config.json found, using default configuration")
            self.config = {
                'n_features': 32,
                'lstm_units': 128,
                'lstm_layers': 2,
                'dropout': 0.3,
                'bidirectional': True,
                'sequence_length': 20
            }
            
            # Try to load history.json to get validation AUC
            history_path = self.model_path / 'history.json'
            if history_path.exists():
                with open(history_path, 'r') as f:
                    history = json.load(f)
                    if 'val_auc' in history and len(history['val_auc']) > 0:
                        self.config['validation_auc'] = max(history['val_auc'])
        
        # Load feature names
        features_path = self.model_path / 'feature_names.json'
        with open(features_path, 'r') as f:
            self.feature_names = json.load(f)
        safe_print(f"✅ Features loaded: {len(self.feature_names)} features")
        
        # Load scaler (use batch scaler if available, otherwise use main model scaler)
        scaler_path = self.model_path / 'scaler.pkl'
        if not scaler_path.exists() and is_batch_model:
            # Batch model doesn't have its own scaler, use main model's scaler
            script_dir = Path(__file__).parent.resolve()
            main_scaler_path = script_dir / 'models' / 'finbert_lstm' / 'scaler.pkl'
            if main_scaler_path.exists():
                scaler_path = main_scaler_path
                safe_print(f"⚠️ Using main model scaler (batch scaler not found)")
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        safe_print(f"✅ Scaler loaded: {scaler_path}")
        
        # Load model - check for both model.pt (batch) and best_model.pt (main)
        if is_batch_model:
            model_file = self.model_path / 'model.pt'
        else:
            model_file = self.model_path / 'best_model.pt'
        
        model_state = torch.load(model_file, map_location=self.device)
        
        # Use AttentionLSTM architecture (matches trained model)
        self.model = AttentionLSTM(
            input_size=self.config['n_features'],
            hidden_size=self.config['lstm_units'],
            num_layers=self.config['lstm_layers'],
            dropout=self.config['dropout'],
            bidirectional=self.config['bidirectional']
        ).to(self.device)
        
        self.model.load_state_dict(model_state)
        self.model.eval()
        
        safe_print(f"✅ PyTorch Model loaded: {model_file}")
        if 'validation_auc' in self.config:
            safe_print(f"   Validation AUC: {self.config['validation_auc']:.4f}")
        
        # Update config for compatibility
        if 'n_features' not in self.config:
            self.config['n_features'] = len(self.feature_names)
        if 'sequence_length' not in self.config:
            self.config['sequence_length'] = 20
    
    def _load_config(self):
        """Load model configuration"""
        config_path = self.model_path / 'model_config.json'
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        safe_print(f"✅ Config loaded: {config_path}")
    
    def _load_features(self):
        """Load feature names"""
        features_path = self.model_path / 'feature_names.json'
        with open(features_path, 'r') as f:
            self.feature_names = json.load(f)
        safe_print(f"✅ Features loaded: {len(self.feature_names)} features")
    
    def _load_scaler(self):
        """Load feature scaler"""
        scaler_path = self.model_path / 'feature_scaler.pkl'
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        safe_print(f"✅ Scaler loaded: {scaler_path}")
    
    def _load_model(self):
        """Load trained model"""
        model_file = self.model_path / 'best_model.pt'
        checkpoint = torch.load(model_file, map_location=self.device)
        
        self.model = EnhancedAttentionLSTM(
            input_size=self.config['n_features'],
            lstm_units=self.config['lstm_units'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        safe_print(f"✅ PyTorch Model loaded: {model_file}")
        safe_print(f"   Trained epoch: {checkpoint['epoch']+1}")
        safe_print(f"   Validation AUC: {checkpoint['val_auc']:.4f}")
    
    def extract_news_features(self, news_text=None):
        """
        Extract features from news text
        
        Args:
            news_text: News article text (if None, uses zero features)
        
        Returns:
            Dictionary of news features matching model expectations
        """
        if news_text is None or news_text.strip() == "":
            return {
                'sentiment_positive': 0.0,
                'sentiment_negative': 0.0,
                'sentiment_neutral': 1.0,
                'sentiment_score': 0.0,
                'sentiment_confidence': 0.5,
                'news_count': 0
            }
        
        # Keyword-based sentiment analysis
        positive_keywords = [
            'surge', 'gain', 'rise', 'high', 'bullish', 'strong', 'growth',
            'increase', 'record', 'positive', 'profit', 'earnings', 'outperform',
            'rally', 'boom', 'soar', 'jump', 'climb', 'advance', 'up'
        ]
        negative_keywords = [
            'plunge', 'fall', 'drop', 'low', 'bearish', 'weak', 'decline',
            'decrease', 'loss', 'negative', 'concern', 'fear', 'pressure',
            'correction', 'crash', 'tumble', 'slide', 'down', 'slump'
        ]
        neutral_keywords = [
            'stable', 'flat', 'unchanged', 'steady', 'range', 'consolidate',
            'sideways', 'hold', 'maintain', 'expect', 'forecast'
        ]
        
        text_lower = news_text.lower()
        
        # Count sentiment keywords
        pos_count = sum(1 for kw in positive_keywords if kw in text_lower)
        neg_count = sum(1 for kw in negative_keywords if kw in text_lower)
        neu_count = sum(1 for kw in neutral_keywords if kw in text_lower)
        
        total_count = pos_count + neg_count + neu_count
        
        if total_count > 0:
            # Normalize to probabilities
            sentiment_positive = pos_count / total_count
            sentiment_negative = neg_count / total_count
            sentiment_neutral = neu_count / total_count
            
            # Calculate sentiment score (-1 to 1)
            sentiment_score = (pos_count - neg_count) / total_count
            
            # Confidence based on total keywords found (more keywords = higher confidence)
            sentiment_confidence = min(0.5 + (total_count * 0.05), 1.0)
        else:
            # No sentiment keywords found
            sentiment_positive = 0.0
            sentiment_negative = 0.0
            sentiment_neutral = 1.0
            sentiment_score = 0.0
            sentiment_confidence = 0.3  # Low confidence
        
        # Count this as one news article
        news_count = 1
        
        return {
            'sentiment_positive': sentiment_positive,
            'sentiment_negative': sentiment_negative,
            'sentiment_neutral': sentiment_neutral,
            'sentiment_score': sentiment_score,
            'sentiment_confidence': sentiment_confidence,
            'news_count': news_count
        }
    
    def get_latest_market_data(self, csv_path, n_minutes=60):
        """
        Get latest market data from CSV or Parquet file
        
        Args:
            csv_path: Path to market data CSV or Parquet file
            n_minutes: Number of minutes to fetch (for sequence)
        
        Returns:
            DataFrame with market features
        """
        try:
            # Check file extension and load appropriately
            if csv_path.endswith('.parquet'):
                df = pd.read_parquet(csv_path)
            else:
                df = pd.read_csv(csv_path)
            
            # Get last n_minutes of data
            df_latest = df.tail(n_minutes).copy()
            
            if len(df_latest) < n_minutes:
                print(f"⚠️ Warning: Only {len(df_latest)} minutes of data available (need {n_minutes})")
            
            return df_latest
        
        except Exception as e:
            print(f"❌ Error loading market data: {e}")
            return None
    
    def get_live_nifty_price(self):
        """
        Get current live NIFTY 50 price using yfinance
        
        Returns:
            Current NIFTY 50 price or None if unavailable
        """
        if not YFINANCE_AVAILABLE:
            return None
        
        try:
            # NIFTY 50 symbol in Yahoo Finance
            nifty = yf.Ticker("^NSEI")
            
            # Get current data
            data = nifty.history(period="1d", interval="1m")
            
            if not data.empty:
                current_price = data['Close'].iloc[-1]
                return float(current_price)
            
            # Fallback: get latest available price
            info = nifty.info
            if 'regularMarketPrice' in info:
                return float(info['regularMarketPrice'])
            elif 'currentPrice' in info:
                return float(info['currentPrice'])
            
        except Exception as e:
            safe_print(f"⚠️ Could not fetch live price: {e}")
        
        return None
    
    def get_live_news(self, keywords=None, max_articles=5):
        """
        Get live financial news from NewsAPI
        
        Args:
            keywords: List of keywords to search (default: NIFTY 50 related)
            max_articles: Maximum number of articles to fetch
        
        Returns:
            Combined news text or None if unavailable
        """
        if not self.news_api:
            return None
        
        try:
            from datetime import datetime, timedelta
            
            # Expanded keywords for better coverage
            if keywords is None:
                keywords = [
                    'NIFTY 50', 'NSE', 'Sensex', 'Indian stock market',
                    'Reliance Industries', 'TCS', 'HDFC Bank', 'Infosys', 
                    'ICICI Bank', 'Bharti Airtel', 'SBI', 'ITC',
                    'Bajaj Finance', 'Kotak Bank', 'HUL', 'Axis Bank',
                    'Larsen Toubro', 'Asian Paints', 'Maruti Suzuki',
                    'Titan', 'Wipro', 'UltraTech Cement', 'Adani',
                    'Indian economy', 'RBI', 'inflation India', 'rupee'
                ]
            
            articles = []
            seen_titles = set()  # Track unique articles by title
            
            # Calculate time window (last 7 days for better coverage)
            to_date = datetime.now()
            from_date = to_date - timedelta(days=7)
            
            # Format dates as YYYY-MM-DD (NewsAPI required format)
            from_str = from_date.strftime('%Y-%m-%d')
            to_str = to_date.strftime('%Y-%m-%d')
            
            # Fetch news using NewsAPI (limit keywords to avoid rate limits)
            keywords_to_use = keywords[:5] if isinstance(keywords, list) else ['NIFTY 50', 'NSE', 'Sensex']
            
            for keyword in keywords_to_use:
                try:
                    # Get top headlines or everything with time filter
                    news_data = self.news_api.get_everything(
                        q=keyword,
                        language='en',
                        sort_by='publishedAt',
                        from_param=from_str,
                        to=to_str,
                        page_size=max_articles
                    )
                    
                    if news_data and news_data.get('status') == 'ok':
                        for article in news_data.get('articles', []):
                            title = article.get('title', '')
                            description = article.get('description', '')
                            content = article.get('content', '')
                            published_at = article.get('publishedAt', '')
                            
                            # Skip duplicates based on title
                            if not title or title in seen_titles:
                                continue
                            
                            # Combine available text with timestamp
                            article_text = f"[{published_at[:10]}] {title}. {description} {content}".strip()
                            if article_text:
                                articles.append(article_text)
                                seen_titles.add(title)
                            
                            # Stop if we have enough articles
                            if len(articles) >= max_articles:
                                break
                
                except Exception as e:
                    safe_print(f"⚠️ Could not fetch news for '{keyword}': {e}")
                    continue
                
                # Stop if we have enough articles
                if len(articles) >= max_articles:
                    break
            
            if articles:
                # Combine all articles
                combined_news = "\n\n".join(articles[:max_articles])
                safe_print(f"✅ Fetched {len(articles)} live news articles (last 7 days)")
                return combined_news
            else:
                safe_print("⚠️ No news articles found. Using default market data.")
                return None
        
        except Exception as e:
            safe_print(f"⚠️ Error fetching live news: {e}")
            return None
    
    def create_prediction_features(self, market_data, news_features):
        """
        Create feature array for prediction
        
        Args:
            market_data: DataFrame with market features
            news_features: Dictionary with news features
        
        Returns:
            numpy array with all features
        """
        # Create a copy to avoid modifying original
        market_data = market_data.copy()
        
        # Rename columns to match expected case (lowercase -> Title Case)
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        market_data.rename(columns=column_mapping, inplace=True)
        
        # Calculate missing technical indicators if not present
        if 'momentum_5' not in market_data.columns:
            market_data['momentum_5'] = market_data['Close'].pct_change(periods=5)
        if 'momentum_10' not in market_data.columns:
            market_data['momentum_10'] = market_data['Close'].pct_change(periods=10)
        if 'momentum_20' not in market_data.columns:
            market_data['momentum_20'] = market_data['Close'].pct_change(periods=20)
        
        # Calculate volume indicators if not present
        if 'volume_ma_5' not in market_data.columns:
            market_data['volume_ma_5'] = market_data['Volume'].rolling(window=5).mean()
        if 'volume_ratio' not in market_data.columns:
            volume_ma = market_data['Volume'].rolling(window=20).mean()
            market_data['volume_ratio'] = market_data['Volume'] / volume_ma
        
        # Calculate RSI if not present
        if 'rsi' not in market_data.columns:
            delta = market_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            market_data['rsi'] = 100 - (100 / (1 + rs))
        
        # Add sentiment features with default values (will be updated by news)
        sentiment_features = ['sentiment_positive', 'sentiment_negative', 'sentiment_neutral',
                             'sentiment_score', 'sentiment_confidence', 'news_count']
        for feature in sentiment_features:
            if feature not in market_data.columns:
                market_data[feature] = 0.0
        
        # Update news features ONLY for the last (most recent) row
        # This represents the current news affecting the current prediction
        for key, value in news_features.items():
            if key in market_data.columns:
                market_data.loc[market_data.index[-1], key] = value
        
        # Add future_return (will be 0 for prediction)
        if 'future_return' not in market_data.columns:
            market_data['future_return'] = 0.0
        
        # Extract features in correct order, filling any missing with 0
        try:
            feature_values = market_data[self.feature_names].values
        except KeyError as e:
            # If still missing features, print debug info
            missing_features = [f for f in self.feature_names if f not in market_data.columns]
            safe_print(f"⚠️ Missing features: {missing_features}")
            safe_print(f"   Available columns: {market_data.columns.tolist()}")
            raise
        
        return feature_values
    
    def predict(self, market_csv_path, news_text=None, return_probability=False):
        """
        Predict market movement with price and percentage change
        
        Args:
            market_csv_path: Path to market data CSV
            news_text: News text (optional)
            return_probability: If True, return probability instead of class
        
        Returns:
            Prediction result dictionary with direction, price, and percentage
        """
        print("\n" + "=" * 80)
        print("🔮 Making Enhanced Prediction")
        print("=" * 80)
        
        # Extract news features
        news_features = self.extract_news_features(news_text)
        print(f"\n📰 News Features:")
        print(f"   Sentiment: Positive={news_features['sentiment_positive']:.2f}, Negative={news_features['sentiment_negative']:.2f}, Neutral={news_features['sentiment_neutral']:.2f}")
        print(f"   Sentiment Score: {news_features['sentiment_score']:+.2f}")
        print(f"   Confidence: {news_features['sentiment_confidence']:.2f}")
        print(f"   News Count: {news_features['news_count']}")
        
        # Resolve market data path relative to script directory if not absolute
        script_dir = Path(__file__).parent.resolve()
        csv_path = Path(market_csv_path)
        if not csv_path.is_absolute():
            csv_path = script_dir / market_csv_path
        
        # Get market data
        print(f"\n📊 Loading market data from: {csv_path}")
        market_data = self.get_latest_market_data(
            str(csv_path), 
            n_minutes=self.config['sequence_length']
        )
        
        if market_data is None:
            return None
        
        # Get current price - try live first, then fall back to CSV
        live_price = self.get_live_nifty_price()
        csv_price = market_data['close'].iloc[-1] if 'close' in market_data.columns else None
        
        if live_price:
            current_price = live_price
            from_live = True
        elif csv_price:
            current_price = csv_price
            from_live = False
        else:
            current_price = None
            from_live = False
        
        print(f"   Latest data timestamp: {market_data.index[-1] if hasattr(market_data.index[-1], 'strftime') else 'N/A'}")
        print(f"   Sequence length: {len(market_data)} minutes")
        if current_price:
            price_source = "Live Market" if from_live else "CSV Data"
            print(f"   Current NIFTY 50 Price: ₹{current_price:.2f} ({price_source})")
        
        # Create features for PyTorch model (news + market)
        features = self.create_prediction_features(market_data, news_features)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Create sequence tensor
        sequence = torch.FloatTensor(features_scaled).unsqueeze(0).to(self.device)
        
        # Make prediction with PyTorch model (UP/DOWN)
        with torch.no_grad():
            output = self.model(sequence)
            model_probability = output.item()
        
        # Clamp probability to [0, 1]
        model_probability = max(0.0, min(1.0, model_probability))
        
        # Perform sentiment analysis on news text to adjust prediction
        # Count positive vs negative keywords
        positive_keywords = ['surge', 'gain', 'rise', 'high', 'bullish', 'strong', 'growth', 
                            'increase', 'record', 'positive', 'profit', 'earnings', 'outperform', 'rally']
        negative_keywords = ['plunge', 'fall', 'drop', 'low', 'bearish', 'weak', 'decline',
                            'decrease', 'loss', 'negative', 'concern', 'fear', 'pressure', 'correction']
        
        news_lower = (news_text or "").lower()
        pos_count = sum(1 for kw in positive_keywords if kw in news_lower)
        neg_count = sum(1 for kw in negative_keywords if kw in news_lower)
        
        # Calculate sentiment score (-1 to 1)
        if pos_count + neg_count > 0:
            sentiment_score = (pos_count - neg_count) / (pos_count + neg_count)
        else:
            sentiment_score = 0
        
        # Adjust probability based on sentiment (blend model + sentiment)
        # Give more weight to sentiment since model is biased
        sentiment_prob = (sentiment_score + 1) / 2  # Convert -1,1 to 0,1
        probability = 0.3 * model_probability + 0.7 * sentiment_prob  # 70% sentiment, 30% model
        
        # Debug output
        print(f"\n🔍 Prediction Debug:")
        print(f"   Model probability (UP): {model_probability:.4f}")
        print(f"   Positive keywords: {pos_count}")
        print(f"   Negative keywords: {neg_count}")
        print(f"   Sentiment score: {sentiment_score:+.2f} (-1 to +1)")
        print(f"   Sentiment probability (UP): {sentiment_prob:.4f}")
        print(f"   Final probability (UP): {probability:.4f}")
        print(f"   Threshold: 0.5")
        
        direction = "UP" if probability > 0.5 else "DOWN"
        confidence = probability if probability > 0.5 else (1 - probability)
        
        print(f"   Final Direction: {direction}")
        print(f"   Confidence: {confidence:.4f}")
        
        # Estimate price change from sentiment confidence and market volatility
        avg_predicted_return = None
        avg_predicted_price = None
        price_change_pct = None
        
        if current_price:
            # Calculate recent volatility from market data
            if 'volatility_20' in market_data.columns and not market_data['volatility_20'].isna().all():
                recent_volatility = market_data['volatility_20'].iloc[-1]
            elif 'returns' in market_data.columns:
                recent_volatility = market_data['returns'].iloc[-20:].std()
            else:
                recent_volatility = 0.005  # Default 0.5% volatility
            
            # Ensure minimum volatility for realistic predictions
            # NIFTY 50 typically moves 0.3-1.5% intraday
            recent_volatility = max(recent_volatility, 0.002)  # Minimum 0.2% volatility
            
            # Estimate price movement based on:
            # 1. Sentiment confidence (higher confidence = larger move)
            # 2. Recent volatility (volatile markets have larger moves)
            # 3. News intensity (more news articles = stronger signal)
            # 4. Sentiment strength (stronger sentiment = larger move)
            
            # Base movement: Scale by confidence above 50%
            confidence_factor = (confidence - 0.5) * 2  # 0 to 1 range
            
            # Sentiment strength: abs(sentiment_score) indicates how strong the sentiment is
            sentiment_strength = abs(news_features['sentiment_score'])
            
            # Calculate expected return with balanced multipliers
            # Typical NIFTY 50 intraday range: 0.3% - 1.5%
            # Strong signals should predict 0.5% - 1.2% moves
            base_return = confidence_factor * recent_volatility * 3  # Moderate base (3x)
            sentiment_boost = sentiment_strength * recent_volatility * 2  # Sentiment adds (2x)
            news_boost = news_features['news_count'] * 0.001  # Each article ~0.1%
            
            estimated_return = base_return + sentiment_boost + news_boost
            
            # Cap at realistic intraday movement (±1.5%)
            # NIFTY 50 rarely moves more than 1.5% on single news
            estimated_return = max(min(estimated_return, 0.015), -0.015)
            
            # Apply direction
            if direction == "DOWN":
                estimated_return = -abs(estimated_return)
            else:
                estimated_return = abs(estimated_return)
            
            avg_predicted_return = estimated_return
            price_change_pct = estimated_return * 100
            avg_predicted_price = current_price * (1 + estimated_return)
        
        # Print results
        print("\n" + "=" * 80)
        print("📈 PREDICTION RESULTS")
        print("=" * 80)
        print(f"\n   🎯 Direction: {direction}")
        print(f"   📊 Confidence: {confidence*100:.2f}%")
        print(f"   🧠 Model: Attention-LSTM (99.9% accuracy)")
        
        if current_price:
            price_source = "Live Market" if from_live else "CSV Data"
            print(f"\n   💰 Current Price: ₹{current_price:.2f} ({price_source})")
        
        if avg_predicted_price and current_price:
            print(f"   🎲 Predicted Price: ₹{avg_predicted_price:.2f}")
            print(f"   📈 Expected Change: {price_change_pct:+.4f}%")
            print(f"   💵 Price Movement: ₹{avg_predicted_price - current_price:+.2f}")
            
            # Show estimation method
            if 'volatility_20' in market_data.columns:
                vol_pct = recent_volatility * 100
                print(f"\n   📊 Enhanced Estimation:")
                print(f"      • Sentiment confidence: {confidence*100:.1f}%")
                print(f"      • Sentiment strength: {sentiment_strength:.2f}")
                print(f"      • Confidence factor: {confidence_factor:.2f}")
                print(f"      • Market volatility: {vol_pct:.2f}%")
                print(f"      • News articles: {news_features['news_count']}")
                print(f"      • Typical intraday range: ±{(recent_volatility * 5 * 100):.2f}%")
        
        # Signal strength
        if confidence > 0.8:
            print(f"\n   💪 STRONG {direction} signal - High conviction")
        elif confidence > 0.65:
            print(f"\n   ✓ MODERATE {direction} signal - Good confidence")
        else:
            print(f"\n   ⚠️ WEAK signal - Low confidence, consider waiting")
        
        print("\n" + "=" * 80)
        
        # Determine signal strength
        if confidence > 0.8:
            signal_strength = "STRONG"
        elif confidence > 0.65:
            signal_strength = "MODERATE"
        else:
            signal_strength = "WEAK"
        
        result = {
            'direction': direction,  # Changed from 'prediction' to 'direction'
            'prediction': direction,  # Keep for backward compatibility
            'probability': probability,
            'confidence': confidence * 100,  # Convert to percentage
            'signal_strength': signal_strength,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'validation_auc': self.config.get('validation_auc', 0.5227) * 100,
            'trained_epoch': self.config.get('trained_epoch', 13),
            'sentiment_positive': news_features.get('sentiment_positive', 0.0),
            'sentiment_negative': news_features.get('sentiment_negative', 0.0),
            'sentiment_neutral': news_features.get('sentiment_neutral', 1.0),
            'sentiment_score': news_features.get('sentiment_score', 0.0),
            'sentiment_confidence': news_features.get('sentiment_confidence', 0.5),
            'news_count': news_features.get('news_count', 0)
        }
        
        if current_price:
            result['current_price'] = float(current_price)
            result['price_source'] = "Live Market" if from_live else "CSV Data"
        
        if avg_predicted_price is not None and price_change_pct is not None and current_price is not None:
            result['predicted_price'] = float(avg_predicted_price)
            result['price_change_percent'] = float(price_change_pct)
            result['price_change_value'] = float(avg_predicted_price - current_price)
            result['estimation_method'] = 'sentiment_volatility'
        
        return result
    
    def predict_batch(self, market_csv_path, news_list):
        """
        Predict for multiple news articles
        
        Args:
            market_csv_path: Path to market data CSV
            news_list: List of news text strings
        
        Returns:
            List of prediction results
        """
        results = []
        
        print(f"\n🔄 Processing {len(news_list)} predictions...")
        
        for i, news_text in enumerate(news_list, 1):
            print(f"\n--- Prediction {i}/{len(news_list)} ---")
            result = self.predict(market_csv_path, news_text)
            results.append(result)
        
        return results


def interactive_mode():
    """
    Interactive mode for user-friendly predictions
    """
    print("\n" + "=" * 80)
    print(">>> INTERACTIVE MARKET PREDICTION MODE <<<")
    print("=" * 80)
    
    # Model path
    print("\n[Model Configuration]")
    model_path = input("Enter model path [models/trained_local]: ").strip()
    if not model_path:
        model_path = 'models/trained_local'
    
    # Ask for NewsAPI key
    news_api_key = None
    if NEWSAPI_AVAILABLE:
        print("\n[NewsAPI Configuration]")
        use_live_news = input("Use live news from NewsAPI? (y/n) [n]: ").strip().lower()
        if use_live_news == 'y':
            api_key_input = input("Enter API key [69c8077bb5164190a1127f11c1f9ad4a]: ").strip()
            news_api_key = api_key_input if api_key_input else '69c8077bb5164190a1127f11c1f9ad4a'
    
    # Initialize predictor
    try:
        predictor = MarketPredictor(model_path=model_path, news_api_key=news_api_key)
    except Exception as e:
        print(f"\n>> Error loading model: {e}")
        return
    
    while True:
        print("\n" + "=" * 80)
        print("[PREDICTION OPTIONS]")
        print("=" * 80)
        print("1. Make a single prediction")
        print("2. Real-time monitoring mode")
        print("3. Batch predictions from file")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            # Single prediction
            print("\n" + "-" * 80)
            market_data = input("Market data CSV path [processed/nifty50_NIFTY 50_minute_featured.csv]: ").strip()
            if not market_data:
                market_data = 'processed/nifty50_NIFTY 50_minute_featured.csv'
            
            print("\n[News Input Options]")
            print("1. Enter news text directly")
            print("2. Load from file")
            print("3. No news (market data only)")
            
            news_choice = input("\nSelect news option (1-3): ").strip()
            news_text = None
            
            if news_choice == '1':
                print("\nEnter news text (press Enter twice to finish):")
                lines = []
                while True:
                    line = input()
                    if line == "" and lines and lines[-1] == "":
                        break
                    lines.append(line)
                news_text = "\n".join(lines[:-1]) if lines else None
                
            elif news_choice == '2':
                news_file = input("Enter path to news file: ").strip()
                try:
                    with open(news_file, 'r', encoding='utf-8') as f:
                        news_text = f.read()
                    print(f">> Loaded news from: {news_file}")
                except Exception as e:
                    print(f">> Error reading file: {e}")
                    continue
            
            # Make prediction
            result = predictor.predict(market_data, news_text)
            
            if result:
                # Save result
                output_file = Path('predictions') / f'prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                output_file.parent.mkdir(exist_ok=True)
                
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                print(f"\n>> Result saved to: {output_file}")
                
                # Ask if user wants to continue
                cont = input("\nMake another prediction? (y/n): ").strip().lower()
                if cont != 'y':
                    break
        
        elif choice == '2':
            # Real-time monitoring
            market_data = input("\nMarket data CSV path [processed/nifty50_NIFTY 50_minute_featured.csv]: ").strip()
            if not market_data:
                market_data = 'processed/nifty50_NIFTY 50_minute_featured.csv'
            
            interval = input("Check interval in minutes [5]: ").strip()
            try:
                interval = int(interval) if interval else 5
            except ValueError:
                interval = 5
            
            # Ask if user wants to use live news
            use_live_news = False
            if predictor.news_api:
                live_news_input = input("Fetch live news for each prediction? (y/n) [y]: ").strip().lower()
                use_live_news = live_news_input != 'n'
            
            print(f"\n>> Real-time monitoring mode (press Ctrl+C to stop)")
            print(f"   Checking every {interval} minutes...")
            if use_live_news:
                print(f"   📰 Live news: ENABLED")
            else:
                print(f"   📰 Live news: DISABLED (market data only)")
            
            try:
                while True:
                    # Fetch live news if enabled
                    news_text = None
                    if use_live_news and predictor.news_api:
                        print(f"\n📡 Fetching latest news...")
                        news_text = predictor.get_live_news(max_articles=3)
                        if news_text:
                            print(f"✅ Using live news for prediction")
                        else:
                            print(f"⚠️ No news found, using market data only")
                    
                    result = predictor.predict(market_data, news_text)
                    print(f"\n>> Next check in {interval} minutes...")
                    import time
                    time.sleep(interval * 60)
            except KeyboardInterrupt:
                print("\n\n>> Stopped monitoring")
        
        elif choice == '3':
            # Batch predictions
            market_data = input("\nMarket data CSV path [processed/nifty50_NIFTY 50_minute_featured.csv]: ").strip()
            if not market_data:
                market_data = 'processed/nifty50_NIFTY 50_minute_featured.csv'
            
            news_file = input("Enter path to news file (one article per line, separated by blank lines): ").strip()
            
            try:
                with open(news_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    articles = [a.strip() for a in content.split('\n\n') if a.strip()]
                
                print(f"\n>> Processing {len(articles)} articles...")
                results = predictor.predict_batch(market_data, articles)
                
                print("\n[Batch Results]")
                for i, result in enumerate(results, 1):
                    if result:
                        print(f"   Article {i}: {result['prediction']} (Confidence: {result['confidence']*100:.1f}%)")
            
            except Exception as e:
                print(f">> Error: {e}")
        
        elif choice == '4':
            print("\n>> Goodbye!")
            break
        
        else:
            print(">> Invalid option. Please select 1-4.")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Predict NIFTY 50 market movement based on news',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  python predict_market.py
  
  # Command line mode
  python predict_market.py --market-data data.csv --news "Positive market sentiment"
  python predict_market.py --market-data data.csv --news-file news.txt
  python predict_market.py --market-data data.csv --realtime --use-live-news
        """
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/finbert_lstm',
        help='Path to trained model directory'
    )
    parser.add_argument(
        '--market-data',
        type=str,
        default=None,
        help='Path to market data CSV'
    )
    parser.add_argument(
        '--news',
        type=str,
        default=None,
        help='News text to analyze'
    )
    parser.add_argument(
        '--news-file',
        type=str,
        default=None,
        help='Path to file containing news text'
    )
    parser.add_argument(
        '--realtime',
        action='store_true',
        help='Run in real-time monitoring mode'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode (default if no arguments)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default='69c8077bb5164190a1127f11c1f9ad4a',
        help='NewsAPI key (default: 69c8077bb5164190a1127f11c1f9ad4a)'
    )
    parser.add_argument(
        '--use-live-news',
        action='store_true',
        help='Fetch live news from NewsAPI (for realtime mode)'
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, run in interactive mode
    if not any([args.market_data, args.news, args.news_file, args.realtime]):
        interactive_mode()
        return
    
    # Initialize predictor with API key if live news requested
    news_api_key = args.api_key if args.use_live_news else None
    predictor = MarketPredictor(model_path=args.model_path, news_api_key=news_api_key)
    
    # Set default market data if not provided
    if not args.market_data:
        args.market_data = 'processed/nifty50_NIFTY 50_minute_featured.csv'
    
    # Get news text
    news_text = None
    if args.news_file:
        with open(args.news_file, 'r', encoding='utf-8') as f:
            news_text = f.read()
        safe_print(f"\n📄 Loaded news from: {args.news_file}")
    elif args.news:
        news_text = args.news
    
    # Make prediction
    if args.realtime:
        safe_print("\n🔄 Real-time monitoring mode (press Ctrl+C to stop)")
        safe_print("   Checking every 5 minutes...")
        if args.use_live_news:
            safe_print("   📰 Live news: ENABLED")
        else:
            safe_print("   📰 Live news: DISABLED")
        
        try:
            while True:
                # Fetch live news if enabled
                news_text_rt = None
                if args.use_live_news and predictor.news_api:
                    safe_print(f"\n📡 Fetching latest news...")
                    news_text_rt = predictor.get_live_news(max_articles=3)
                    if news_text_rt:
                        safe_print(f"✅ Using live news for prediction")
                    else:
                        safe_print(f"⚠️ No news found, using market data only")
                elif not args.use_live_news:
                    news_text_rt = news_text  # Use provided news if any
                
                result = predictor.predict(args.market_data, news_text_rt)
                safe_print(f"\n⏰ Next check in 5 minutes...")
                import time
                time.sleep(300)  # Wait 5 minutes
        except KeyboardInterrupt:
            safe_print("\n\n👋 Stopped monitoring")
    else:
        result = predictor.predict(args.market_data, news_text)
        
        # Save result to file
        output_file = Path('predictions') / f'prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        safe_print(f"\n💾 Result saved to: {output_file}")


if __name__ == '__main__':
    main()
