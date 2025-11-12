"""
Model Loading Utilities for FinBERT+LSTM
=========================================
Provides functions to load the trained FinBERT+LSTM model for predictions.
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import pickle


class AttentionLSTM(nn.Module):
    """
    Attention-LSTM Model Architecture
    Matches the architecture used during training (Batch 3: 78.27% AUC)
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


def load_finbert_lstm_model(model_dir='finbert_lstm', device='cpu'):
    """
    Load the trained FinBERT+LSTM model (Best: Batch 3 with 78.27% AUC)
    
    Args:
        model_dir: Directory containing model files (relative to models/)
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        Dictionary containing:
        - model: Loaded PyTorch model
        - scaler: Fitted RobustScaler
        - feature_names: List of feature names
        - config: Model configuration
    """
    # Construct full path
    if not Path(model_dir).is_absolute():
        script_dir = Path(__file__).parent.resolve()
        model_dir = script_dir / model_dir
    else:
        model_dir = Path(model_dir)
    
    device = torch.device(device)
    
    # Load configuration
    with open(model_dir / 'config.json', 'r') as f:
        config = json.load(f)
    
    # Load feature names
    with open(model_dir / 'feature_names.json', 'r') as f:
        feature_names = json.load(f)
    
    # Load scaler
    with open(model_dir / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Initialize model
    model = AttentionLSTM(
        input_size=config['n_features'],
        hidden_size=config['lstm_units'],
        num_layers=config['lstm_layers'],
        dropout=config['dropout'],
        bidirectional=config['bidirectional']
    ).to(device)
    
    # Load model weights
    model_state = torch.load(model_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(model_state)
    model.eval()
    
    print(f"✅ Model loaded from: {model_dir}")
    print(f"   Features: {len(feature_names)}")
    print(f"   Validation AUC: {config.get('validation_auc', 'N/A'):.4f}")
    print(f"   Model: {config.get('active_model', 'Best available')}")
    print(f"   Description: {config.get('model_description', 'N/A')}")
    
    return {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'config': config,
        'device': device
    }


def load_batch_model(batch_idx, model_dir='finbert_lstm', device='cpu'):
    """
    Load a specific batch model
    
    Args:
        batch_idx: Batch number (1-5)
        model_dir: Base model directory
        device: Device to load model on
    
    Returns:
        Dictionary containing model, scaler, features, config
    """
    # Construct full path
    if not Path(model_dir).is_absolute():
        script_dir = Path(__file__).parent.resolve()
        model_dir = script_dir / model_dir
    else:
        model_dir = Path(model_dir)
    
    # Find batch directory
    batch_dirs = list(model_dir.glob(f'batch_{batch_idx}_*'))
    if not batch_dirs:
        raise FileNotFoundError(f"Batch {batch_idx} model not found in {model_dir}")
    
    batch_dir = batch_dirs[0]
    
    # Load configuration from main directory
    with open(model_dir / 'config.json', 'r') as f:
        config = json.load(f)
    
    # Load batch-specific files
    with open(batch_dir / 'feature_names.json', 'r') as f:
        feature_names = json.load(f)
    
    with open(batch_dir / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Initialize and load model
    device = torch.device(device)
    model = AttentionLSTM(
        input_size=len(feature_names),
        hidden_size=config['lstm_units'],
        num_layers=config['lstm_layers'],
        dropout=config['dropout'],
        bidirectional=config['bidirectional']
    ).to(device)
    
    model_state = torch.load(batch_dir / 'model.pt', map_location=device)
    model.load_state_dict(model_state)
    model.eval()
    
    # Load batch history for AUC info
    try:
        with open(batch_dir / 'history.json', 'r') as f:
            history = json.load(f)
            best_auc = max(history['val_auc'])
            print(f"✅ Batch {batch_idx} model loaded from: {batch_dir.name}")
            print(f"   Best Validation AUC: {best_auc:.4f}")
    except:
        print(f"✅ Batch {batch_idx} model loaded from: {batch_dir.name}")
    
    return {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'config': config,
        'device': device,
        'batch_dir': batch_dir
    }


# Example usage:
if __name__ == '__main__':
    # Load best model (Batch 3)
    model_data = load_finbert_lstm_model()
    print(f"\n✅ Ready for predictions!")
    print(f"   Input features: {len(model_data['feature_names'])}")
    print(f"   Sequence length: {model_data['config']['sequence_length']}")
    
    # Load specific batch
    # batch3 = load_batch_model(3)
    # print(f"\n✅ Batch 3 model loaded with {batch3['config']['n_features']} features")
