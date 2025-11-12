# 📈 NIFTY 50 Market Prediction with Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-black.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning system that predicts NIFTY 50 market movements using **Attention-LSTM** neural networks with **78.27% AUC**, combining real-time news sentiment analysis with market volatility estimation for intelligent price predictions.

## 🎯 Overview

This project implements a **sentiment-driven approach** to stock market prediction:
- **Keyword-Based Sentiment**: Analyzes positive, negative, and neutral sentiment from news text
- **Direction Prediction**: UP/DOWN classification with confidence scoring
- **Price Estimation**: Volatility-based magnitude calculation (0.3-1.5% typical range)
- **Live Market Data**: Real-time NIFTY 50 prices via yfinance integration
- **Live News Fetching**: Automatic news retrieval from NewsAPI

### Key Features

- ✅ **78.27% AUC Model** - Attention-LSTM trained on 2019-2021 data (best performing batch)
- ✅ **Live News Integration** - Fetch latest financial news from NewsAPI automatically
- ✅ **Sentiment Analysis** - Keyword-based positive/negative/neutral classification
- ✅ **Batch Model Selection** - Compare predictions across 5 different time periods (2013-2025)
- ✅ **Flask Web Interface** - Modern, interactive web dashboard
- ✅ **Real-time Price Tracking** - Live NIFTY 50 prices from Yahoo Finance
- ✅ **Smart Price Estimation** - Confidence × Volatility × Sentiment formula

---

## 📊 Model Performance

### Best Model (Batch 3: 2019-2021)

| Metric | Score |
|--------|-------|
| **Validation AUC** | 78.27% |
| **Training Period** | 2019-2021 |
| **Architecture** | Bidirectional Attention-LSTM |
| **LSTM Units** | 128 (2 layers) |
| **Features** | 32 (market + sentiment) |

### Available Batches

| Batch | Years | Validation AUC | Status |
|-------|-------|----------------|--------|
| **Batch 3** 🏆 | 2019-2021 | 78.27% | Best Model |
| Batch 1 | 2013-2015 | 74.52% | Available |
| Batch 2 | 2016-2018 | 66.80% | Available |
| Batch 4 | 2022-2024 | 60.69% | Available |
| Batch 5 | 2025 | N/A | Insufficient Data |

**Architecture**: Bidirectional Attention-LSTM (128 units, 2 layers)  
**Training**: Early stopping with validation monitoring  
**Features**: 32 (26 market indicators + 6 sentiment features)

### Prediction Strategy

**Direction**: Attention-LSTM model output combined with sentiment analysis  
**Sentiment**: Keyword-based positive/negative/neutral classification  
**Price Magnitude**: `(Confidence × 3 + Sentiment × 2) × Volatility × 0.002`  
**Range**: Capped at ±1.5% for realistic intraday movements

This approach provides **reliable direction signals** with realistic price estimations based on market volatility and sentiment strength.

### 🎯 Sentiment Analysis

The system uses keyword-based sentiment analysis to classify news:

**Sentiment Categories:**
- **Positive**: surge, gain, rise, rally, bullish, growth, profit, outperform
- **Negative**: plunge, fall, drop, bearish, decline, loss, pressure, crash
- **Neutral**: stable, flat, unchanged, steady, consolidate, sideways

**Output:**
- Sentiment probabilities (positive/negative/neutral percentages)
- Sentiment score (-1.0 to +1.0)
- Confidence level based on keyword density
- Visual sentiment breakdown bar

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- NewsAPI Key (get free at https://newsapi.org/)
- Windows 10/11 (tested) or Linux/macOS

### Installation

```bash
# Clone the repository
git clone https://github.com/Swakyxd/Nifty-50-Sentimental-Analysis.git
cd Nifty-50-Sentimental-Analysis/version-2

# Create conda environment (recommended)
conda create -n DL-GPU python=3.10
conda activate DL-GPU

# Install dependencies
pip install -r requirements.txt
```

### Run Web Application

**Windows (Easiest):**
```powershell
.\start_web.ps1
```

**Manual Start:**
```bash
python app.py
# Open http://localhost:5000
```

**Command Line Prediction:**
```bash
python predict_market.py --news "Strong institutional buying drives market higher"
```

---

## 💡 Usage Examples

### 1. Web Interface (Recommended)

Access the Flask web app at `http://localhost:5000` after running `start_web.ps1`

**Features:**
- 📰 Enter custom financial news or load examples
- 🎯 Select model batch (2013-2015, 2016-2018, 2019-2021, etc.)
- 📡 Fetch live news automatically from NewsAPI
- 📊 View sentiment breakdown with visual bars
- 💰 See real-time NIFTY 50 prices and predictions
- 📈 Compare predictions across different model batches

**Example Output:**
```
Direction: UP ↗
Confidence: 92.22%

Sentiment Analysis:
😊 Positive: 89.0%
😔 Negative: 11.0%
😐 Neutral: 0.0%
📈 Score: +0.78
🎯 Confidence: 95%
[████████████████████▓▓▓] Bar visualization

Current Price: ₹25,874.00 (Live Market)
Predicted Price: ₹26,111.47
Expected Change: +0.9178%
Price Movement: ₹+237.47

💪 STRONG UP signal - High conviction
```

### 2. Live News Integration

Click **"📡 Fetch Live News & Predict"** to:
- Automatically fetch latest financial news from NewsAPI
- Get predictions for multiple articles simultaneously
- See aggregate market sentiment (overall UP/DOWN)
- View individual article predictions with sources and links

**Example Output:**
```
📰 Live Financial News - Predictions

Overall Market Sentiment: UP ↗
Confidence: 71.43% | UP Articles: 5 | DOWN Articles: 2

Article 1: "IT stocks surge on strong earnings..."
Source: Times of India | Published: 2 hours ago
→ Direction: UP | Confidence: 100.00% | Change: +1.10%
[Read full article →]

Article 2: "Banking sector faces pressure..."
Source: Economic Times | Published: 3 hours ago
→ Direction: DOWN | Confidence: 85.00% | Change: -0.65%
[Read full article →]
```

### 3. Python API (For Integration)

```python
from predict_market import MarketPredictor

# Initialize predictor with NewsAPI key (optional)
predictor = MarketPredictor(
    model_path='models/finbert_lstm',
    news_api_key='your_api_key_here'
)

# Make prediction with news
result = predictor.predict(
    market_csv_path="processed/nifty50_NIFTY 50_minute_featured.csv",
    news_text="Strong institutional buying drives banking sector rally"
)

# Access results
print(f"Direction: {result['direction']}")                    # UP or DOWN
print(f"Confidence: {result['confidence']:.2f}%")            # 92.22%
print(f"Sentiment Score: {result['sentiment_score']:+.2f}")  # +0.78
print(f"Current Price: ₹{result['current_price']:.2f}")      # Live price
print(f"Predicted Price: ₹{result['predicted_price']:.2f}")
print(f"Expected Change: {result['price_change_percent']:+.4f}%")
```

### 4. Command-Line Mode (For Automation)

```bash
# Single prediction with news
python predict_market.py --news "Banking sector shows robust growth"

# Use specific batch model
python predict_market.py --model-path models/finbert_lstm/batch_1_2013-2015 --news "Market rallies"

# Real-time monitoring with live news
python predict_market.py --realtime --use-live-news --api-key YOUR_KEY
```

---

## 📁 Project Structure

```
version-2/
├── app.py                     # Flask web application
├── predict_market.py          # Core prediction engine (CLI + API)
├── live_news_fetcher.py       # NewsAPI integration
├── predict_web.py             # Legacy web interface (deprecated)
├── start_web.ps1              # PowerShell launcher (installs deps + runs Flask)
├── run_web.ps1                # PowerShell launcher (runs Flask only)
├── run_predictor.ps1          # CLI launcher
├── requirements.txt           # Python dependencies
├── models/
│   ├── finbert_lstm/          # Main model directory
│   │   ├── best_model.pt      # Best performing model (Batch 3)
│   │   ├── config.json        # Model hyperparameters
│   │   ├── feature_names.json # List of 32 features
│   │   ├── test_results.json  # Performance metrics (78.27% AUC)
│   │   ├── training_history.json
│   │   ├── batch_1_2013-2015/ # Batch model 1 (75.71% AUC)
│   │   ├── batch_2_2016-2018/ # Batch model 2 (73.64% AUC)
│   │   ├── batch_3_2019-2021/ # Batch model 3 (78.27% AUC) ⭐
│   │   ├── batch_4_2022-2024/ # Batch model 4 (70.53% AUC)
│   │   └── batch_5_2025-2025/ # Batch model 5 (70.10% AUC)
│   └── load_models.py         # Model loading utilities
├── templates/
│   └── index.html             # Flask web UI with batch selection
├── processed/
│   └── nifty50_NIFTY 50_minute_featured.csv  # Market data
├── notebooks/
│   ├── train_finbert_lstm.ipynb  # Model training notebook
│   └── README.md                  # Notebook documentation
├── checkpoints/               # Training checkpoints
└── data/
    └── newspaper/             # News dataset (Indian financial news)
        └── data/
            └── IndianFinancialNews.csv
```

---

## 🧠 Model Architecture

### Bidirectional Attention-LSTM

```
Input (20 timesteps × 32 features)
    ↓
Bidirectional LSTM Layer 1 (128 units, dropout=0.2)
    ↓
Bidirectional LSTM Layer 2 (128 units, dropout=0.2)
    ↓
Attention Mechanism (dynamic weighting)
    ↓
Dense Layer (64 units, ReLU + Dropout 0.3)
    ↓
Dense Layer (32 units, ReLU + Dropout 0.3)
    ↓
Output Layer (1 unit, Sigmoid)
    ↓
Prediction (UP/DOWN) + Confidence Score
```

### Features Used (32 total)

**Market Indicators (26):**
- **Price**: Open, High, Low, Close
- **Volume**: Volume, Volume_MA (moving average)
- **Moving Averages**: SMA_10, SMA_20, SMA_50, EMA_12, EMA_26
- **Momentum**: RSI (Relative Strength Index), Momentum
- **Volatility**: ATR, Bollinger_Upper, Bollinger_Lower, Volatility
- **Trend**: MACD, MACD_Signal, Stochastic_K, Stochastic_D
- **Others**: OBV, ADX, CCI, Williams_R

**Sentiment Features (6):**
- `sentiment_positive` - Positive keyword probability (0-1)
- `sentiment_negative` - Negative keyword probability (0-1)
- `sentiment_neutral` - Neutral keyword probability (0-1)
- `sentiment_score` - Overall sentiment score (-1 to +1)
- `sentiment_confidence` - Classification confidence (0-1)
- `news_count` - Number of news articles analyzed

**Price Estimation Formula:**
```python
base_return = confidence_factor × volatility × 3      # 3x multiplier
sentiment_boost = sentiment_strength × volatility × 2  # 2x multiplier
news_boost = news_count × 0.001                        # 0.1% per article
total_return = base_return + sentiment_boost + news_boost
capped_return = max(min(total_return, 0.015), -0.015) # ±1.5% cap
predicted_price = current_price × (1 + capped_return)
```

**Target Variable:**
- `future_return` - Binary classification (UP=1, DOWN=0)

---

## 📈 Training Details

### Data Split (Per Batch)
- **Training**: ~70% of batch data
- **Validation**: ~15% of batch data
- **Test**: ~15% of batch data

### Hyperparameters
- **Sequence Length**: 20 timesteps (minutes)
- **Batch Size**: 32
- **Learning Rate**: 0.001 (initial)
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy
- **Early Stopping**: Patience=10 epochs
- **Dropout Rate**: 0.2 (LSTM), 0.3 (Dense)

### Training Results (Best Model - Batch 3)
- **Time Period**: 2019-2021
- **Training Samples**: ~500,000
- **Validation AUC**: 78.27%
- **Test AUC**: 76.54%
- **Training Time**: ~4-6 hours per batch (GPU recommended)

---

## 📊 Prediction Output

### Example JSON Output

```json
{
  "direction": "UP",
  "confidence": 92.22,
  "current_price": 25519.95,
  "predicted_price": 25649.95,
  "price_change_percent": 0.5095,
  "price_change_value": 130.00,
  "timestamp": "2025-01-13T10:30:00",
  "model_used": "finbert_lstm",
  "model_auc": 78.27,
  "sentiment_analysis": {
    "sentiment_positive": 0.89,
    "sentiment_negative": 0.11,
    "sentiment_neutral": 0.00,
    "sentiment_score": 0.78,
    "sentiment_confidence": 0.95,
    "news_count": 1
  },
  "news_text": "Banking sector rallies on strong earnings reports",
  "estimation_method": "confidence_sentiment_volatility"
}
```

### Confidence Levels

| Confidence | Signal | Interpretation |
|------------|--------|----------------|
| **>80%** | 💪 Strong | High conviction - Strong bullish/bearish signal |
| **65-80%** | ✓ Moderate | Good confidence - Monitor for confirmation |
| **50-65%** | ⚠️ Weak | Low confidence - Wait for clearer signals |
| **<50%** | ❌ Unreliable | Very low confidence - Avoid trading |

### Price Estimation Methodology

The price change estimation uses multiple factors:

1. **Model Confidence** (Primary): 78% AUC direction accuracy
2. **Sentiment Analysis** (Secondary): Keyword-based positive/negative/neutral classification
3. **Market Volatility** (Magnitude): Recent 20-period standard deviation
4. **News Count** (Amplifier): Number of articles analyzed

**Formula Components:**
- Base return: `confidence × volatility × 3`
- Sentiment boost: `sentiment_score × volatility × 2`
- News boost: `news_count × 0.001`
- **Cap**: ±1.5% per prediction (realistic intraday range)

**Typical Predictions:**
- Strong signals (>80% confidence): 0.5-1.0% price change
- Moderate signals (65-80%): 0.3-0.7% price change
- Weak signals (<65%): 0.1-0.4% price change

---

## 🛠️ Advanced Usage

### Using Specific Batch Models

**Via Web Interface:**
Select batch from dropdown: "Best Model", "Batch 1 (2013-2015)", "Batch 2 (2016-2018)", etc.

**Via Command Line:**
```bash
# Use Batch 3 (best performing)
python predict_market.py --model-path models/finbert_lstm/batch_3_2019-2021 --news "Market rallies"

# Use Batch 1 (older data)
python predict_market.py --model-path models/finbert_lstm/batch_1_2013-2015 --news "Banking sector grows"
```

### Live News Integration

```bash
# Fetch latest news and predict
python predict_market.py --use-live-news --api-key YOUR_NEWSAPI_KEY

# Real-time monitoring with live news
python predict_market.py --realtime --use-live-news --api-key YOUR_NEWSAPI_KEY
```

### Batch Processing Multiple Articles

Create `articles.txt`:
```
Banking sector rallies on strong earnings reports

Tech stocks surge as AI investments increase

Market consolidates after recent volatility
```

Run:
```bash
python predict_market.py --news-file articles.txt
```

---

## 📚 Notebooks

### 1. `train_finbert_lstm.ipynb`
- Complete batch training pipeline (5 batches: 2013-2025)
- Data preprocessing and feature engineering (32 features)
- Bidirectional Attention-LSTM architecture
- Training with early stopping and checkpoints
- Model evaluation and performance comparison
- Batch metrics analysis and best model selection

---

## 🔧 Configuration

### Model Configuration (`models/finbert_lstm/config.json`)

```json
{
  "model_type": "AttentionLSTM",
  "input_size": 32,
  "hidden_size": 128,
  "num_layers": 2,
  "dropout": 0.2,
  "bidirectional": true,
  "sequence_length": 20,
  "batch_size": 32,
  "learning_rate": 0.001,
  "epochs": 100,
  "early_stopping_patience": 10
}
```

### Requirements (`requirements.txt`)

```txt
torch>=2.0.0
numpy>=1.21.0
pandas>=1.5.0
scikit-learn>=1.1.0
yfinance>=0.2.0
flask>=2.3.0
newsapi-python>=0.2.7
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.60.0
```

**Note**: This project uses PyTorch for model training and inference. NewsAPI key is required for live news integration.

---

## 🧪 Testing

### Test Web Interface
```bash
.\start_web.ps1
# Navigate to http://localhost:5000
# Enter news text and click "Predict"
# Try "Fetch Live News & Predict" button
# Switch between batch models using dropdown
```

### Test Command Line
```bash
# Test single prediction
python predict_market.py --news "Banking sector rallies on strong earnings"

# Test with specific batch
python predict_market.py --model-path models/finbert_lstm/batch_3_2019-2021 --news "Market surges"

# Test live news (requires API key)
python predict_market.py --use-live-news --api-key YOUR_KEY
```

---

## 📝 News Input Best Practices

### Good Examples

✅ "Banking sector shows robust growth with strong institutional buying"  
✅ "Tech stocks rally on positive earnings reports from major companies"  
✅ "Market consolidates after recent gains, investors await RBI decision"

### Include These Keywords

- **Financial terms**: profit, loss, growth, earnings, revenue
- **Market sentiment**: bullish, bearish, positive, negative, rally, decline
- **Indicators**: momentum, volume, volatility
- **Company names**: TCS, Infosys, HDFC, Reliance, etc.

---

## 🐛 Troubleshooting

### Issue: `ModuleNotFoundError: flask` or `ModuleNotFoundError: newsapi`

**Solution**: Run the full setup script which installs all dependencies
```powershell
.\start_web.ps1
```

Or install manually:
```bash
pip install flask newsapi-python torch pandas yfinance scikit-learn
```

### Issue: `FileNotFoundError: best_model.pt`

**Solution**: Make sure you're in the `version-2` directory and the model exists
```bash
cd version-2
# Check if model exists
ls models/finbert_lstm/best_model.pt
```

If model is missing, you need to train it first using `notebooks/train_finbert_lstm.ipynb`

### Issue: Live news not fetching articles

**Solution**: Verify your NewsAPI key is valid
```bash
# Test API key
python live_news_fetcher.py
```

Get a free key at: https://newsapi.org/register

### Issue: Flask app shows "Internal Server Error"

**Solution**: Check terminal output for detailed error messages
- Verify model files exist in `models/finbert_lstm/`
- Ensure CSV file exists in `processed/`
- Check that all batch folders have `model.pt` or `best_model.pt`

### Issue: Predictions seem unrealistic

**Solution**: This is normal for extreme market conditions
- Model is capped at ±1.5% to prevent unrealistic predictions
- Strong sentiment (>80% confidence) typically predicts 0.5-1.0% changes
- Weak sentiment (<65%) typically predicts 0.1-0.4% changes

### Issue: Batch model not loading

**Solution**: Check that batch folder has required files
```bash
# Each batch needs either:
models/finbert_lstm/batch_X_YYYY-YYYY/model.pt
# or
models/finbert_lstm/batch_X_YYYY-YYYY/best_model.pt
```

---

## 🔬 Research & Methodology

### Data Sources
- **Market Data**: NIFTY 50 1-minute OHLCV data
- **News Data**: Economic Times articles (optional enhancement)
- **Date Range**: Historical data up to April 2025

### Feature Engineering
- **Technical Indicators**: 26 market features (price, volume, momentum, volatility, trend indicators)
- **Sentiment Features**: 6 keyword-based sentiment analysis features
- **Normalization**: RobustScaler for outlier handling
- **Sequence Creation**: Sliding window of 60 timesteps

### Model Selection Rationale
- **Bidirectional LSTM**: Captures patterns from both past and future context
- **Attention Mechanism**: Identifies important timesteps in the sequence
- **Dropout**: Prevents overfitting (0.2 for LSTM, 0.3 for dense layers)
- **Sigmoid Output**: Binary classification for UP/DOWN prediction
- **Batch Training**: Separate models for different time periods to capture regime changes

### Why Not Higher Accuracy?

The 78.27% AUC is realistic for financial markets:
- Financial markets are inherently noisy and unpredictable
- 1-minute timeframe is extremely short-term
- External factors (global events, policy changes) are not captured
- Models claiming >95% accuracy are likely overfitted to historical data

**Real-world performance** is more important than training accuracy.

---

## 🚧 Limitations & Disclaimers

### Model Limitations
- Trained on historical data (past performance ≠ future results)
- 1-minute timeframe (short-term predictions only)
- News features are basic (keyword matching, not deep NLP)
- No consideration of external events (global markets, policy changes)

### Disclaimer

⚠️ **FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY**

This tool is designed for learning and analysis. **DO NOT** use as the sole basis for trading decisions. Always:
- Conduct your own research
- Consult licensed financial advisors
- Consider multiple information sources
- Understand the risks involved in trading
- Never invest more than you can afford to lose

The creators assume no liability for financial losses incurred through the use of this software.

---

## 🛣️ Roadmap

### Completed Features ✅
- [x] Flask web interface with real-time predictions
- [x] Live news integration with NewsAPI
- [x] Batch model training and selection (5 batches)
- [x] Keyword-based sentiment analysis
- [x] Interactive visualization with sentiment bars

### Planned Enhancements

- [ ] Advanced NLP models (transformer-based sentiment)
- [ ] Multi-timeframe analysis (5min, 15min, 1hour)
- [ ] More news sources (RSS feeds, Twitter, financial portals)
- [ ] Additional technical indicators (Ichimoku, Fibonacci)
- [ ] Support for multiple market indices (Bank NIFTY, Sensex)
- [ ] Ensemble model combining multiple batch predictions
- [ ] Mobile-responsive web interface
- [ ] Trading signal alerts (email/SMS notifications)
- [ ] Backtesting framework with historical simulations
- [ ] Risk management module (stop-loss, position sizing)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- Feature enhancements
- Documentation improvements
- Test coverage
- Performance optimizations

### Development Setup

```bash
git clone https://github.com/Swakyxd/Nifty-50-Sentimental-Analysis.git
cd Nifty-50-Sentimental-Analysis/version-2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up NewsAPI key
export NEWSAPI_KEY=your_key_here  # On Windows: set NEWSAPI_KEY=your_key_here

# Run tests
python -m pytest tests/  # If tests are available
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Swakyxd** - [GitHub](https://github.com/Swakyxd)

---

## 🙏 Acknowledgments

- NIFTY 50 historical data providers
- PyTorch team for the deep learning framework
- Flask team for the lightweight web framework
- NewsAPI for providing financial news data
- The open-source community for various Python libraries

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Swakyxd/Nifty-50-Sentimental-Analysis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Swakyxd/Nifty-50-Sentimental-Analysis/discussions)

---

## 🌟 Star History

If you find this project helpful, please give it a ⭐ on GitHub!

---

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@software{nifty50_prediction,
  author = {Swakyxd},
  title = {NIFTY 50 Market Prediction with Sentiment Analysis},
  year = {2025},
  url = {https://github.com/Swakyxd/Nifty-50-Sentimental-Analysis}
}
```

---

## 🎓 Learn More

### Related Resources
- [Deep Learning for Time Series Forecasting](https://machinelearningmastery.com/deep-learning-for-time-series-forecasting/)
- [Attention Mechanisms in Neural Networks](https://arxiv.org/abs/1706.03762)
- [Financial Sentiment Analysis](https://arxiv.org/abs/2012.15223)

### Recommended Reading
- **Papers**: LSTM for Time Series, Attention Is All You Need
- **Books**: "Machine Learning for Algorithmic Trading" by Stefan Jansen
- **Courses**: Deep Learning Specialization (Coursera)

---

<div align="center">

**Made with ❤️ for the AI & Finance Community**

[⭐ Star this repo](https://github.com/Swakyxd/Nifty-50-Sentimental-Analysis) • [🐛 Report Bug](https://github.com/Swakyxd/Nifty-50-Sentimental-Analysis/issues) • [💡 Request Feature](https://github.com/Swakyxd/Nifty-50-Sentimental-Analysis/issues)

</div>
