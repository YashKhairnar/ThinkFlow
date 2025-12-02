"""
Test Script for Seq2Seq Medium LSTM Model
Load and test the trained model on EEG data
"""

import sys
import os
sys.path.append('src')

import torch
import numpy as np

from src.seq2seq_model import Seq2Seq
from src.vocabulary import Vocabulary
from src.data_loader import DataLoader
from src.utils import calculate_word_accuracy, calculate_word_error_rate

def load_model(checkpoint_path='checkpoints/seq2seq_medium_model.pth'):
    """Load the trained Seq2Seq model"""

    print("\n" + "="*70)
    print("LOADING SEQ2SEQ MEDIUM MODEL")
    print("="*70)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Display info
    size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"✓ File: {checkpoint_path}")
    print(f"✓ Size: {size_mb:.2f} MB")
    print(f"✓ Epoch: {checkpoint.get('epoch', 'N/A') + 1}")
    print(f"✓ Val Accuracy: {checkpoint.get('val_acc', 'N/A')}%")
    print(f"✓ Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")

    # Get parameters
    vocabulary = checkpoint['vocabulary']
    downsample_factor = checkpoint.get('downsample_factor', 2)

    print(f"✓ Vocabulary size: {len(vocabulary.word2idx)}")
    print(f"✓ Downsample factor: {downsample_factor}")

    # Initialize model
    device = torch.device('cpu')
    sequence_length_downsampled = 5500 // downsample_factor

    model = Seq2Seq(
        vocab_size=len(vocabulary.word2idx),
        input_channels=105,
        sequence_length=sequence_length_downsampled,
        embedding_dim=256,
        encoder_hidden_size=256,
        decoder_hidden_size=256,
        num_layers=2,
        dropout=0.3
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Model loaded successfully!")
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("="*70)

    return model, vocabulary, downsample_factor, device


def predict_single_file(model, vocabulary, downsample_factor, device,
                        eeg_data, true_text=None):
    """
    Predict sentence from EEG data

    Args:
        model: Loaded Seq2Seq model
        vocabulary: Vocabulary object
        downsample_factor: EEG downsampling factor
        device: torch device
        eeg_data: EEG data array (105, 5500)
        true_text: Optional ground truth text

    Returns:
        predicted_text: Predicted sentence
        metrics: Dictionary with accuracy metrics (if true_text provided)
    """

    # Downsample
    eeg_downsampled = eeg_data[:, ::downsample_factor]

    # Prepare input
    eeg_tensor = torch.FloatTensor(eeg_downsampled).unsqueeze(0).to(device)

    # Generate prediction
    with torch.no_grad():
        predicted_ids, _ = model.generate(eeg_tensor, max_len=50)
        predicted_text = vocabulary.decode(predicted_ids[0].cpu().tolist())

    # Calculate metrics if true text provided
    metrics = None
    if true_text is not None:
        word_acc, correct_words, total_words = calculate_word_accuracy(true_text, predicted_text)
        wer, ops = calculate_word_error_rate(true_text, predicted_text)

        metrics = {
            'word_accuracy': word_acc,
            'correct_words': correct_words,
            'total_words': total_words,
            'wer': wer,
            'sentence_match': (true_text.lower().strip() == predicted_text.lower().strip())
        }

    return predicted_text, metrics


def demo():
    """Quick demo of model usage"""

    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*15 + "SEQ2SEQ LSTM MODEL - DEMO" + " "*28 + "█")
    print("█" + " "*68 + "█")
    print("█"*70)

    # Load model
    model, vocabulary, downsample_factor, device = load_model()

    # Check if data directory exists
    data_dir = 'processed_data'
    if not os.path.exists(data_dir):
        print(f"\n⚠️  Data directory '{data_dir}' not found!")
        print("   Place your processed_data folder here to test the model.")
        print("\n" + "█"*70 + "\n")
        return

    # Load data
    print("\n" + "="*70)
    print("LOADING TEST DATA")
    print("="*70)

    data_loader = DataLoader(data_dir)
    data_loader.load_mapping()

    # Test files from the trained sentences
    test_files = [
        'rawdata_6127.csv',
        'rawdata_6254.csv',
        'rawdata_5968.csv',
        'rawdata_6046.csv',
        'rawdata_6124.csv',
    ]

    print(f"✓ Testing on {len(test_files)} samples")

    # Test each file
    print("\n" + "="*70)
    print("PREDICTIONS")
    print("="*70)

    results = []

    for i, filename in enumerate(test_files, 1):
        print(f"\n[{i}/{len(test_files)}] {filename}")
        print("-" * 70)

        # Load data
        eeg_data = data_loader.load_padded_data(filename, target_length=5500)
        true_text = data_loader.get_text_for_file(filename)

        if eeg_data is None:
            print("  ❌ Failed to load data")
            continue

        # Predict
        predicted_text, metrics = predict_single_file(
            model, vocabulary, downsample_factor, device,
            eeg_data, true_text
        )

        # Display
        true_display = true_text[:60] + "..." if len(true_text) > 60 else true_text
        pred_display = predicted_text[:60] + "..." if len(predicted_text) > 60 else predicted_text

        print(f"True:      {true_display}")
        print(f"Predicted: {pred_display}")
        print(f"")
        print(f"Match:     {'✓ CORRECT' if metrics['sentence_match'] else '✗ WRONG'}")
        print(f"Word Acc:  {metrics['word_accuracy']*100:.1f}% ({metrics['correct_words']}/{metrics['total_words']})")
        print(f"WER:       {metrics['wer']*100:.1f}%")

        results.append(metrics)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    total = len(results)
    correct = sum(1 for r in results if r['sentence_match'])
    avg_word_acc = sum(r['word_accuracy'] for r in results) / total if total > 0 else 0
    avg_wer = sum(r['wer'] for r in results) / total if total > 0 else 0

    print(f"Total Samples:         {total}")
    print(f"Correct Sentences:     {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"Average Word Accuracy: {avg_word_acc*100:.1f}%")
    print(f"Average WER:           {avg_wer*100:.1f}%")
    print("="*70)

    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*25 + "DEMO COMPLETE" + " "*30 + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")


if __name__ == "__main__":
    demo()
