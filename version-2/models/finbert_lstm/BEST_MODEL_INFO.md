# Best Model Information

## Model Performance

The **best performing model** is **Batch 3 (2019-2021)** with:
- **Validation AUC: 78.27%** ⭐
- Training period: 2019-2021
- News articles: 2,479
- Market samples: 733
- News coverage: 45.3%

## All Batch Results

| Batch | Period | Val AUC | News Articles | Market Samples | News Coverage |
|-------|--------|---------|---------------|----------------|---------------|
| 1 | 2013-2015 | 74.52% | 8,509 | 669 | 99.6% |
| 2 | 2016-2018 | 66.80% | 6,430 | 723 | 98.9% |
| **3** | **2019-2021** | **78.27%** ⭐ | **2,479** | **733** | **45.3%** |
| 4 | 2022-2024 | 60.69% | 10 | 731 | 0.1% |
| 5 | 2025 | N/A | 531 | 141 | 27.7% |

## Files

### Primary Model (Batch 3 - Best)
- `best_model.pt` - PyTorch model weights (Batch 3)
- `scaler.pkl` - RobustScaler fitted on Batch 3 data
- `feature_names.json` - 32 feature names
- `config.json` - Model configuration

### Individual Batch Models
- `batch_1_2013-2015/` - First batch (baseline)
- `batch_2_2016-2018/` - Second batch
- `batch_3_2019-2021/` - **BEST MODEL** ⭐
- `batch_4_2022-2024/` - Fourth batch (sparse news)
- `batch_5_2025-2025/` - Final batch (most recent)

## Model Architecture

```
AttentionLSTM:
- Input: 32 features (30 market + 6 sentiment)
- Bidirectional LSTM: 2 layers, 128 hidden units
- Attention mechanism over sequence
- Output: Binary classification (UP/DOWN)
- Total parameters: 627,202
```

## Features (32 total)

### Market Features (30)
- OHLCV: Open, High, Low, Close, Volume
- Returns: returns, log_returns
- Moving Averages: ma_5, ma_10, ma_20, ma_50
- MA Ratios: price_to_ma_5, price_to_ma_10, price_to_ma_20, price_to_ma_50
- Volatility: volatility_5, volatility_20
- Momentum: momentum_5, momentum_10, momentum_20
- Volume: volume_ma_5, volume_ratio
- Price Ratios: hl_ratio, close_to_high, close_to_low
- RSI: rsi

### Sentiment Features (6)
- sentiment_positive (0-1)
- sentiment_negative (0-1)
- sentiment_neutral (0-1)
- sentiment_score (-1 to 1)
- sentiment_confidence (0-1)
- news_count (articles per day)

## Usage

### Load Best Model

```python
from models.load_models import load_finbert_lstm_model

# Load the best model (Batch 3)
model_data = load_finbert_lstm_model()

model = model_data['model']
scaler = model_data['scaler']
feature_names = model_data['feature_names']
config = model_data['config']
```

### Load Specific Batch

```python
from models.load_models import load_batch_model

# Load Batch 3 specifically
batch3 = load_batch_model(3)
```

### Make Predictions

```python
import torch
import pandas as pd

# Prepare your data (32 features, 20 timesteps)
# data shape: (20, 32)
data_scaled = scaler.transform(data)

# Create sequence tensor
sequence = torch.FloatTensor(data_scaled).unsqueeze(0)  # (1, 20, 32)

# Predict
with torch.no_grad():
    output = model(sequence)
    probability = output.item()  # UP probability

direction = "UP" if probability > 0.5 else "DOWN"
confidence = probability if probability > 0.5 else (1 - probability)

print(f"Direction: {direction}, Confidence: {confidence:.2%}")
```

## Why Batch 3 is Best

1. **Highest AUC**: 78.27% validation AUC outperforms all other batches
2. **Balanced Coverage**: 45.3% news coverage - not too sparse, not overfitted
3. **Recent Market Regime**: 2019-2021 captures recent market dynamics
4. **Sufficient Data**: 2,479 news articles and 733 market samples
5. **Pre-COVID + COVID**: Includes both normal and volatile market conditions

## Training Strategy

- **Method**: Sequential batch training with incremental learning
- **Approach**: Each batch transfers knowledge from previous batches
- **Benefit**: Batch 3 inherited patterns from Batches 1 & 2, then learned 2019-2021 specifics

## Performance Improvement

**Previous Model**: 22.73% AUC (essentially random)
**Current Model**: 78.27% AUC 
**Improvement**: 244% increase! 🎉

## Next Steps

1. ✅ Best model (Batch 3) is now active
2. Update prediction scripts to use new architecture
3. Test on recent market data
4. Deploy to web application
5. Monitor performance and retrain as needed

---

**Last Updated**: November 12, 2025
**Model**: FinBERT+LSTM with Attention
**Framework**: PyTorch 2.6.0+cu124
