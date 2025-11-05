"""
NIFTY 50 Market Prediction Script
==================================
Predicts market movement (UP/DOWN) based on news sentiment and market data.

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
import sys
warnings.filterwarnings('ignore')

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
    Attention-LSTM Model for Market Prediction
    """
    def __init__(self, input_size, lstm_units=128, dropout=0.2):
        super(AttentionLSTM, self).__init__()
        self.lstm_units = lstm_units
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_units, 
                           batch_first=True, dropout=dropout)
        self.attention = nn.Sequential(nn.Linear(lstm_units, 1), nn.Tanh())
        self.dense = nn.Linear(lstm_units, lstm_units)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(lstm_units, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_scores = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_scores, dim=1)
        attended = lstm_out * attention_weights
        context = torch.sum(attended, dim=1)
        context = self.dense(context)
        context = torch.relu(context)
        context = self.dropout(context)
        output = self.output(context)
        output = self.sigmoid(output)
        return output


class MarketPredictor:
    """
    Market Movement Predictor
    """
    
    def __init__(self, model_path='models/trained_local'):
        # Get script directory and construct absolute path
        script_dir = Path(__file__).parent.resolve()
        self.model_path = script_dir / model_path if not Path(model_path).is_absolute() else Path(model_path)
        self.device = torch.device('cpu')  # Use CPU for prediction
        
        safe_print("=" * 80)
        safe_print("🚀 NIFTY 50 Market Predictor")
        safe_print("=" * 80)
        
        # Load configuration
        self._load_config()
        
        # Load feature names
        self._load_features()
        
        # Load scaler
        self._load_scaler()
        
        # Load model
        self._load_model()
        
        safe_print("✅ Predictor initialized successfully")
        safe_print("=" * 80)
    
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
        
        self.model = AttentionLSTM(
            input_size=self.config['n_features'],
            lstm_units=self.config['lstm_units'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        safe_print(f"✅ Model loaded: {model_file}")
        safe_print(f"   Trained epoch: {checkpoint['epoch']+1}")
        safe_print(f"   Validation AUC: {checkpoint['val_auc']:.4f}")
    
    def extract_news_features(self, news_text=None):
        """
        Extract features from news text
        
        Args:
            news_text: News article text (if None, uses zero features)
        
        Returns:
            Dictionary of news features
        """
        if news_text is None or news_text.strip() == "":
            return {
                'news_num_companies': 0,
                'news_num_keywords': 0,
                'news_text_length': 0
            }
        
        # Simple keyword detection (you can enhance this)
        keywords = [
            'profit', 'loss', 'growth', 'decline', 'market', 'stock',
            'revenue', 'earnings', 'bullish', 'bearish', 'rally', 'crash',
            'positive', 'negative', 'increase', 'decrease', 'nifty', 'sensex'
        ]
        
        text_lower = news_text.lower()
        num_keywords = sum(1 for keyword in keywords if keyword in text_lower)
        
        # Estimate number of companies mentioned (simplified)
        company_indicators = ['ltd', 'limited', 'corp', 'inc', 'industries']
        num_companies = sum(1 for ind in company_indicators if ind in text_lower)
        
        return {
            'news_num_companies': num_companies,
            'news_num_keywords': num_keywords,
            'news_text_length': len(news_text)
        }
    
    def get_latest_market_data(self, csv_path, n_minutes=60):
        """
        Get latest market data from CSV file
        
        Args:
            csv_path: Path to market data CSV
            n_minutes: Number of minutes to fetch (for sequence)
        
        Returns:
            DataFrame with market features
        """
        try:
            df = pd.read_csv(csv_path)
            
            # Get last n_minutes of data
            df_latest = df.tail(n_minutes).copy()
            
            if len(df_latest) < n_minutes:
                print(f"⚠️ Warning: Only {len(df_latest)} minutes of data available (need {n_minutes})")
            
            return df_latest
        
        except Exception as e:
            print(f"❌ Error loading market data: {e}")
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
        # Add news features to market data
        for key, value in news_features.items():
            market_data[key] = value
        
        # Add future_return (will be 0 for prediction)
        market_data['future_return'] = 0.0
        
        # Extract features in correct order
        feature_values = market_data[self.feature_names].values
        
        return feature_values
    
    def predict(self, market_csv_path, news_text=None, return_probability=False):
        """
        Predict market movement
        
        Args:
            market_csv_path: Path to market data CSV
            news_text: News text (optional)
            return_probability: If True, return probability instead of class
        
        Returns:
            Prediction result dictionary
        """
        print("\n" + "=" * 80)
        print("🔮 Making Prediction")
        print("=" * 80)
        
        # Extract news features
        news_features = self.extract_news_features(news_text)
        print(f"\n📰 News Features:")
        print(f"   Companies mentioned: {news_features['news_num_companies']}")
        print(f"   Keywords found: {news_features['news_num_keywords']}")
        print(f"   Text length: {news_features['news_text_length']}")
        
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
        
        print(f"   Latest data timestamp: {market_data.index[-1] if hasattr(market_data.index[-1], 'strftime') else 'N/A'}")
        print(f"   Sequence length: {len(market_data)} minutes")
        
        # Create features
        features = self.create_prediction_features(market_data, news_features)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Create sequence tensor
        sequence = torch.FloatTensor(features_scaled).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(sequence)
            probability = output.item()
        
        # Clamp probability to [0, 1]
        probability = max(0.0, min(1.0, probability))
        
        prediction = "UP" if probability > 0.5 else "DOWN"
        confidence = probability if probability > 0.5 else (1 - probability)
        
        # Print results
        print("\n" + "=" * 80)
        print("📈 PREDICTION RESULTS")
        print("=" * 80)
        print(f"\n   🎯 Prediction: {prediction}")
        print(f"   📊 Probability: {probability:.4f}")
        print(f"   ✅ Confidence: {confidence*100:.2f}%")
        
        if probability > 0.7:
            print(f"   💪 Strong signal for {prediction}")
        elif probability > 0.6:
            print(f"   ✓ Moderate signal for {prediction}")
        else:
            print(f"   ⚠️ Weak signal - Low confidence")
        
        print("\n" + "=" * 80)
        
        return {
            'prediction': prediction,
            'probability': probability,
            'confidence': confidence,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'news_features': news_features
        }
    
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
    
    # Initialize predictor
    try:
        predictor = MarketPredictor(model_path=model_path)
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
            
            print(f"\n>> Real-time monitoring mode (press Ctrl+C to stop)")
            print(f"   Checking every {interval} minutes...")
            
            try:
                while True:
                    result = predictor.predict(market_data, None)
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
  python predict_market.py --market-data data.csv --realtime
        """
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/trained_local',
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
    
    args = parser.parse_args()
    
    # If no arguments provided, run in interactive mode
    if not any([args.market_data, args.news, args.news_file, args.realtime]):
        interactive_mode()
        return
    
    # Initialize predictor
    predictor = MarketPredictor(model_path=args.model_path)
    
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
        
        try:
            while True:
                result = predictor.predict(args.market_data, news_text)
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
