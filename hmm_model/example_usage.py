"""
Example Usage of EEG-to-Text HMM Models

This script demonstrates how to use the pre-trained CNN and HMM models
for EEG-to-text prediction in your own project.
"""

import torch
import numpy as np
from src.feature_extractor import SupervisedCNNEncoder
from src.predictor import SentencePredictor
from src.data_loader import DataLoader
from src.config import *


def example_1_load_models():
    """Example 1: Load pre-trained models"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Loading Pre-trained Models")
    print("="*70)

    # Load CNN encoder
    encoder = SupervisedCNNEncoder(
        input_channels=CNN_INPUT_CHANNELS,
        hidden_channels=CNN_HIDDEN_CHANNELS,
        num_classes=NUM_CLASSES,
        sequence_length=SEQUENCE_LENGTH
    )

    checkpoint = torch.load('checkpoints/cnn_encoder.pth', map_location='cpu')
    encoder.load_state_dict(checkpoint['model_state_dict'])
    encoder.eval()

    print(f"✓ CNN Encoder loaded")
    print(f"  - Input: ({CNN_INPUT_CHANNELS}, {SEQUENCE_LENGTH})")
    print(f"  - Output: ({CNN_HIDDEN_CHANNELS}, ~688)")

    # Load HMM predictor
    predictor = SentencePredictor(
        n_states=HMM_N_STATES,
        n_features=HMM_N_FEATURES
    )
    predictor.load('checkpoints/hmm_models.pkl')

    print(f"✓ HMM Predictor loaded")
    print(f"  - Number of sentence models: {len(predictor.models)}")

    return encoder, predictor


def example_2_extract_features(encoder):
    """Example 2: Extract features from EEG data using CNN"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Extract Features from EEG Data")
    print("="*70)

    # Simulate EEG data (replace with your actual data)
    # Shape: (batch_size, 105 channels, 5500 time steps)
    eeg_data = torch.randn(1, 105, 5500)

    print(f"Input EEG data shape: {eeg_data.shape}")

    # Extract features
    with torch.no_grad():
        features = encoder.get_features(eeg_data)

    print(f"Extracted features shape: {features.shape}")
    print(f"✓ Features extracted successfully")

    return features


def example_3_predict_sentence(predictor, features):
    """Example 3: Predict sentence from features using HMM"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Predict Sentence from Features")
    print("="*70)

    # Convert to numpy and normalize
    features_np = features.cpu().numpy()[0].T  # Shape: (time_steps, 32)

    # Simple standardization
    features_normalized = (features_np - features_np.mean(axis=0)) / (features_np.std(axis=0) + 1e-8)

    print(f"Features shape for HMM: {features_normalized.shape}")

    # Get top 1 prediction
    predicted_text, confidence = predictor.predict(features_normalized)

    print(f"\nPrediction:")
    print(f"  Text: {predicted_text[:80]}...")
    print(f"  Confidence: {confidence:.2f}")

    # Get top 3 predictions
    print(f"\nTop 3 Predictions:")
    top_3 = predictor.predict(features_normalized, top_k=3)
    for i, (text, score) in enumerate(top_3, 1):
        text_preview = text[:60] + "..." if len(text) > 60 else text
        print(f"  {i}. [{score:.2f}] {text_preview}")

    return predicted_text


def example_4_complete_pipeline():
    """Example 4: Complete pipeline from EEG data to text"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Complete Pipeline (EEG → Text)")
    print("="*70)

    # 1. Load models
    print("\n1. Loading models...")
    encoder = SupervisedCNNEncoder(
        input_channels=CNN_INPUT_CHANNELS,
        hidden_channels=CNN_HIDDEN_CHANNELS,
        num_classes=NUM_CLASSES,
        sequence_length=SEQUENCE_LENGTH
    )
    checkpoint = torch.load('checkpoints/cnn_encoder.pth', map_location='cpu')
    encoder.load_state_dict(checkpoint['model_state_dict'])
    encoder.eval()

    predictor = SentencePredictor(n_states=HMM_N_STATES, n_features=HMM_N_FEATURES)
    predictor.load('checkpoints/hmm_models.pkl')
    print("   ✓ Models loaded")

    # 2. Load EEG data (or use simulated data)
    print("\n2. Loading EEG data...")
    # For this example, we'll use random data
    # In practice, load from file: loader.load_padded_data('rawdata_1234.csv')
    eeg_data = torch.randn(1, 105, 5500)
    print(f"   ✓ EEG data loaded: {eeg_data.shape}")

    # 3. Extract features
    print("\n3. Extracting features with CNN...")
    with torch.no_grad():
        features = encoder.get_features(eeg_data)
    features_np = features.cpu().numpy()[0].T
    print(f"   ✓ Features extracted: {features_np.shape}")

    # 4. Normalize features
    print("\n4. Normalizing features...")
    features_normalized = (features_np - features_np.mean(axis=0)) / (features_np.std(axis=0) + 1e-8)
    print(f"   ✓ Features normalized")

    # 5. Predict sentence
    print("\n5. Predicting sentence with HMM...")
    predicted_text, confidence = predictor.predict(features_normalized)
    print(f"   ✓ Prediction complete")

    # 6. Display results
    print("\n" + "-"*70)
    print("FINAL RESULT:")
    print("-"*70)
    print(f"Predicted Text: {predicted_text}")
    print(f"Confidence: {confidence:.2f}")
    print("-"*70)


def example_5_with_real_data():
    """Example 5: Using real EEG data from CSV files"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Using Real EEG Data Files")
    print("="*70)

    # Check if processed_data folder exists
    import os
    if not os.path.exists('processed_data'):
        print("⚠️  'processed_data' folder not found!")
        print("   This example requires the processed_data folder with EEG CSV files.")
        print("   Skipping this example.")
        return

    # Load models
    encoder = SupervisedCNNEncoder(
        input_channels=CNN_INPUT_CHANNELS,
        hidden_channels=CNN_HIDDEN_CHANNELS,
        num_classes=NUM_CLASSES,
        sequence_length=SEQUENCE_LENGTH
    )
    checkpoint = torch.load('checkpoints/cnn_encoder.pth', map_location='cpu')
    encoder.load_state_dict(checkpoint['model_state_dict'])
    encoder.eval()

    predictor = SentencePredictor(n_states=HMM_N_STATES, n_features=HMM_N_FEATURES)
    predictor.load('checkpoints/hmm_models.pkl')

    # Load data
    loader = DataLoader('processed_data')
    loader.load_mapping()

    # Get first file
    files = loader.get_all_files()
    if len(files) == 0:
        print("No data files found!")
        return

    test_file = files[0]
    print(f"Testing with file: {test_file}")

    # Load and process
    eeg_data = loader.load_padded_data(test_file, target_length=SEQUENCE_LENGTH)
    true_text = loader.get_text_for_file(test_file)

    # Extract features
    X = torch.tensor(np.array([eeg_data]), dtype=torch.float32)
    with torch.no_grad():
        features = encoder.get_features(X)
    features_np = features.cpu().numpy()[0].T

    # Normalize and predict
    features_normalized = (features_np - features_np.mean(axis=0)) / (features_np.std(axis=0) + 1e-8)
    predicted_text, confidence = predictor.predict(features_normalized)

    # Display results
    print("\n" + "-"*70)
    print(f"True Text:      {true_text}")
    print(f"Predicted Text: {predicted_text}")
    print(f"Confidence:     {confidence:.2f}")
    print(f"Match:          {'✓ CORRECT' if true_text == predicted_text else '✗ INCORRECT'}")
    print("-"*70)


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("EEG-TO-TEXT MODEL - USAGE EXAMPLES")
    print("="*70)

    # Example 1: Load models
    encoder, predictor = example_1_load_models()

    # Example 2: Extract features
    features = example_2_extract_features(encoder)

    # Example 3: Predict sentence
    example_3_predict_sentence(predictor, features)

    # Example 4: Complete pipeline
    example_4_complete_pipeline()

    # Example 5: Real data (if available)
    example_5_with_real_data()

    print("\n" + "="*70)
    print("✓ All examples completed!")
    print("="*70)
    print()


if __name__ == "__main__":
    main()
