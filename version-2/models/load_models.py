# Model Loading Script for Hybrid CNN-LSTM Models
import tensorflow as tf
import pickle
import json
import numpy as np
from pathlib import Path

def load_models(models_path="models"):
    """Load all trained models and configurations"""
    models_path = Path(models_path)

    # Load configuration
    with open(models_path / "model_configuration.json", 'r') as f:
        config = json.load(f)

    # Load data preparator
    with open(models_path / "data_preparator.pkl", 'rb') as f:
        data_prep = pickle.load(f)

    # Load performance results
    with open(models_path / "model_performance.json", 'r') as f:
        performance = json.load(f)

    # Load models
    models = {}
    model_files = list(models_path.glob("*_model.keras"))
    for model_file in model_files:
        model_name = model_file.stem.replace("_1min_model", "")
        models[model_name] = tf.keras.models.load_model(model_file)
        print(f"Loaded {model_name} (Test Loss: {performance[model_name]['test_loss']:.6f})")

    return models, data_prep, config, performance

def predict_with_model(model, data_prep, new_data, feature_columns):
    """Make predictions with a trained model"""
    # Scale the features
    features_scaled = data_prep.feature_scaler.transform(new_data[feature_columns])

    # Create sequences
    sequences = []
    for i in range(len(features_scaled) - data_prep.sequence_length + 1):
        sequences.append(features_scaled[i:i+data_prep.sequence_length])

    X = np.array(sequences)

    # Make predictions
    predictions = model.predict(X)

    return predictions

# Example usage:
# models, data_prep, config, performance = load_models("models")
# print(f"Available models: {list(models.keys())}")
# print(f"Best model: {min(performance.keys(), key=lambda x: performance[x]['test_loss'])}")
