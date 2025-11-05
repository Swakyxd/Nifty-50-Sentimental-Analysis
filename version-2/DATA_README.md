# Data and Models

Due to GitHub's file size limitations, the following files are not included in this repository:

## Required Data Files

1. **Market Data** (`processed/nifty50_NIFTY 50_minute_featured.csv`)
   - Download from: [Link to be added]
   - Size: ~500MB
   - Place in: `version-2/processed/`

2. **News Data** (`newspaper/data/nifty50_news_extracted.csv`)
   - Download from: [Link to be added]
   - Size: ~50MB
   - Place in: `newspaper/data/`

3. **Trained Models**:
   - `models/trained_local/best_model.pt` - PyTorch model checkpoint
   - `models/trained_local/feature_scaler.pkl` - Feature scaling parameters
   - Download from: [Link to be added]
   - Place in: `version-2/models/trained_local/`

## Alternative: Train Your Own Model

Follow the instructions in `notebooks/train_model_local.ipynb` to:
1. Prepare your own data
2. Train the model from scratch
3. Generate the required files

## Setup Instructions

After downloading the required files:

```bash
# 1. Clone the repository
git clone https://github.com/Swakyxd/Nifty-50-Sentimental-Analysis.git
cd Nifty-50-Sentimental-Analysis/version-2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data files and place them in appropriate directories

# 4. Run predictions
python predict_market.py
```

For detailed instructions, see the main [README.md](README.md).
