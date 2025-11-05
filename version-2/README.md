# 📈 NIFTY 50 Market Prediction with Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.51.0-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning system that predicts NIFTY 50 market movements using Attention-LSTM neural networks, combining technical indicators with news sentiment analysis.

## 🎯 Overview

This project implements a hybrid approach to stock market prediction by combining:
- **Technical Analysis**: 31 market indicators from 1-minute OHLCV data
- **Sentiment Analysis**: News article keyword extraction and company mentions
- **Deep Learning**: Attention-LSTM architecture for sequence modeling

### Key Features

- ✅ **99.25% Test AUC** - High-performance trained model
- ✅ **94.40% Accuracy** on 256K+ test samples
- ✅ **Interactive CLI** - User-friendly menu-driven interface
- ✅ **Web Dashboard** - Streamlit-based visual interface
- ✅ **Real-time Monitoring** - Continuous market prediction
- ✅ **Batch Processing** - Analyze multiple news articles
- ✅ **Python API** - Easy integration with trading systems

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Test AUC** | 99.25% |
| **Test Accuracy** | 94.40% |
| **Precision** | 91.90% |
| **Recall** | 98.42% |
| **Test Samples** | 256,387 |

**Architecture**: Attention-LSTM (128 units)  
**Training**: 20 epochs, Early stopping with patience=5  
**Validation AUC**: 99.70%

---

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- Conda (recommended) or pip
- Windows 10/11 (tested) or Linux/macOS

### Installation

```bash
# Clone the repository
git clone https://github.com/Swakyxd/Nifty-50-Sentimental-Analysis.git
cd Nifty-50-Sentimental-Analysis/version-2

# Create conda environment
conda create -n market-prediction python=3.13
conda activate market-prediction

# Install dependencies
pip install -r requirements.txt
```

### Run Prediction

**Windows (Easiest):**
```bash
run_predictor.bat
```

**Command Line:**
```bash
python predict_market.py --news "Strong institutional buying drives market higher"
```

**Web Interface:**
```bash
streamlit run predict_web.py
# Open http://localhost:8501
```

---

## 💡 Usage Examples

### 1. Interactive Mode (Recommended for Beginners)

```bash
python predict_market.py
```

Follow the menu prompts:
1. Select prediction type (single/batch/real-time)
2. Enter news text or load from file
3. Get instant predictions with confidence scores

### 2. Command-Line Mode (For Automation)

```bash
# Single prediction with news
python predict_market.py --news "Banking sector shows robust growth"

# Real-time monitoring
python predict_market.py --realtime

# Batch processing
python predict_market.py --news-file articles.txt
```

### 3. Python API (For Integration)

```python
from predict_market import MarketPredictor

# Initialize predictor
predictor = MarketPredictor()

# Make prediction
result = predictor.predict(
    market_csv_path="processed/nifty50_NIFTY 50_minute_featured.csv",
    news_text="Positive market momentum observed"
)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']*100:.2f}%")
```

### 4. Web Dashboard

```bash
streamlit run predict_web.py
```

Features:
- Interactive prediction form
- Batch analysis tools
- Real-time monitoring dashboard
- Visual confidence indicators

---

## 📁 Project Structure

```
version-2/
├── predict_market.py          # Main prediction script (CLI + Interactive)
├── predict_web.py             # Streamlit web interface
├── example_predictions.py     # Usage examples
├── run_predictor.bat          # Windows launcher
├── run_predictor.ps1          # PowerShell launcher
├── requirements.txt           # Python dependencies
├── models/
│   └── trained_local/         # Trained model files
│       ├── best_model.pt      # PyTorch model weights
│       ├── feature_scaler.pkl # RobustScaler for normalization
│       ├── feature_names.json # List of 35 features
│       ├── model_config.json  # Model hyperparameters
│       └── test_results.json  # Performance metrics
├── processed/
│   └── nifty50_NIFTY 50_minute_featured.csv  # Market data
├── notebooks/
│   ├── train_model_local.ipynb           # Model training
│   ├── market_prediction_analysis.ipynb  # Analysis
│   └── market_sentiment_analysis.ipynb   # Sentiment features
├── predictions/               # Output JSON files
└── data/
    └── newspaper/             # News data (optional)
```

---

## 🧠 Model Architecture

### Attention-LSTM

```
Input (60 timesteps × 35 features)
    ↓
LSTM Layer (128 units, dropout=0.3)
    ↓
Attention Mechanism
    ↓
Dense Layer (128 units, ReLU)
    ↓
Dropout (0.3)
    ↓
Output Layer (Sigmoid)
    ↓
Prediction (UP/DOWN)
```

### Features Used (35 total)

**Market Indicators (31):**
- Price: Open, High, Low, Close
- Volume indicators
- Moving averages (SMA, EMA)
- Momentum (RSI, MACD, Stochastic)
- Volatility (Bollinger Bands, ATR)
- Trend indicators

**News Sentiment (3):**
- `news_num_companies` - Company mentions count
- `news_num_keywords` - Financial keyword frequency
- `news_text_length` - Article length

**Target (1):**
- `future_return` - Binary classification (UP=1, DOWN=0)

---

## 📈 Training Details

### Data Split
- **Training**: 71.9% (791,639 samples)
- **Validation**: 4.8% (52,974 samples)
- **Test**: 23.3% (256,387 samples)

### Hyperparameters
- **Sequence Length**: 60 minutes
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy
- **Early Stopping**: Patience=5 epochs

### Training Results
- **Best Epoch**: 20
- **Training Time**: ~2 hours (CPU)
- **Final Train Loss**: 0.0156
- **Final Val Loss**: 0.0087
- **Val AUC**: 99.70%

---

## 📊 Prediction Output

### Example JSON Output

```json
{
  "prediction": "UP",
  "probability": 0.9523,
  "confidence": 0.9523,
  "timestamp": "2025-11-05 12:15:04",
  "news_features": {
    "news_num_companies": 2,
    "news_num_keywords": 8,
    "news_text_length": 215
  }
}
```

### Confidence Levels

| Confidence | Signal | Interpretation |
|------------|--------|----------------|
| **>70%** | 💪 Strong | High confidence - Consider action |
| **60-70%** | ⚠️ Moderate | Monitor closely |
| **<60%** | ⚠️ Weak | Low confidence - Wait for better signal |

---

## 🛠️ Advanced Usage

### Custom Model Path

```bash
python predict_market.py --model-path custom/model/path --news "News text"
```

### Batch Predictions

Create `articles.txt`:
```
Article 1: Banking sector rallies...

Article 2: Tech stocks surge...

Article 3: Market consolidates...
```

Run:
```bash
python predict_market.py --news-file articles.txt
```

### Real-Time Monitoring

```bash
python predict_market.py --realtime
# Predictions every 5 minutes
# Press Ctrl+C to stop
```

---

## 📚 Notebooks

### 1. `train_model_local.ipynb`
- Complete training pipeline
- Data preprocessing and feature engineering
- Model architecture implementation
- Training with checkpoints
- Evaluation and visualization

### 2. `market_prediction_analysis.ipynb`
- Exploratory data analysis
- Feature importance analysis
- Model comparison (CNN-LSTM, Attention-LSTM, hybrid)
- Performance benchmarking

### 3. `market_sentiment_analysis.ipynb`
- News data processing
- Sentiment feature extraction
- Keyword analysis
- Company mention detection

---

## 🔧 Configuration

### Model Configuration (`models/trained_local/model_config.json`)

```json
{
  "model_type": "Attention-LSTM",
  "sequence_length": 60,
  "n_features": 35,
  "lstm_units": 128,
  "dropout": 0.3,
  "batch_size": 64,
  "learning_rate": 0.001,
  "epochs": 50,
  "patience": 5
}
```

### Requirements (`requirements.txt`)

```txt
torch>=2.0.0
numpy>=1.21.0
pandas>=1.5.0
scikit-learn>=1.1.0
streamlit>=1.20.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.60.0
```

---

## 🧪 Testing

Run example predictions:

```bash
python example_predictions.py
```

This will demonstrate:
1. Basic prediction with news
2. Market-only prediction
3. Batch predictions (3 articles)
4. Real-time analysis with trading signals

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

### Issue: `ModuleNotFoundError`

**Solution**: Use the batch launcher which includes the correct Python path
```bash
run_predictor.bat
```

Or activate conda environment:
```bash
conda activate market-prediction
python predict_market.py
```

### Issue: `FileNotFoundError: model_config.json`

**Solution**: Make sure you're in the `version-2` directory
```bash
cd version-2
python predict_market.py
```

### Issue: Emojis not displaying in Windows console

**Solution**: This is normal. The script automatically handles encoding issues. All functionality works correctly even if emojis don't display.

### Issue: Out of memory during prediction

**Solution**: Model uses CPU by default. If issues persist:
- Close other applications
- Process news articles in smaller batches
- Reduce sequence length in config (requires retraining)

---

## 🔬 Research & Methodology

### Data Sources
- **Market Data**: NIFTY 50 1-minute OHLCV data
- **News Data**: Economic Times articles (optional enhancement)
- **Date Range**: Historical data up to April 2025

### Feature Engineering
- **Technical Indicators**: 31 features using TA-Lib methods
- **Normalization**: RobustScaler for outlier handling
- **Sequence Creation**: Sliding window of 60 timesteps

### Model Selection Rationale
- **Attention Mechanism**: Captures important time steps in sequence
- **LSTM**: Handles long-term dependencies in time series
- **Dropout**: Prevents overfitting (0.3 rate)
- **Sigmoid Output**: Binary classification for UP/DOWN

---

## 📊 Confusion Matrix

```
                Predicted
              DOWN    UP
Actual DOWN  104,080  12,155  (89.5% specificity)
Actual UP      2,215 137,937  (98.4% recall)
```

**Interpretation**:
- Model catches 98.4% of upward movements (high recall)
- Low false negatives (2,215) - rarely misses UP movements
- Some false positives (12,155) - occasionally predicts UP when market goes DOWN

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

### Planned Enhancements

- [ ] Advanced NLP models (BERT, FinBERT) for sentiment
- [ ] Multi-timeframe analysis (5min, 15min, 1hour)
- [ ] Integration with real-time news APIs
- [ ] Additional technical indicators
- [ ] Support for multiple market indices
- [ ] Ensemble model with multiple architectures
- [ ] Mobile app interface
- [ ] Trading bot integration (with safety features)
- [ ] Backtesting framework
- [ ] Risk management module

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
conda create -n market-prediction-dev python=3.13
conda activate market-prediction-dev
pip install -r requirements.txt
pip install pytest black flake8  # Development tools
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
- Streamlit for the web interface framework
- The open-source community

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
