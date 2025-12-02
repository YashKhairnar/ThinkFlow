# EEG-to-Text Seq2Seq LSTM Model Package

This is a self-contained package for using a pre-trained Seq2Seq LSTM model for EEG-to-text prediction.

## Contents

```
lstm_model/
├── checkpoints/
│   └── seq2seq_medium_model.pth    # Trained model (17.32 MB)
├── src/
│   ├── seq2seq_model.py            # Seq2Seq model architecture
│   ├── vocabulary.py               # Vocabulary and tokenization
│   ├── config_lstm.py              # Configuration parameters
│   ├── data_loader.py              # Data loading utilities
│   └── utils.py                    # Utility functions
├── test_lstm_model.py              # Test/demo script
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```

## Model Information

- **Architecture**: Encoder-Decoder LSTM with Bahdanau Attention
- **Model Size**: 17.32 MB
- **Parameters**: 4,533,801
- **Vocabulary Size**: 72 words
- **Training Accuracy**: 40% (validation)
- **Word-Level Accuracy**: ~61%
- **Input**: EEG data (105 channels × 5500 timesteps)
- **Output**: Text sentence (word-by-word generation)

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pandas

Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Run Demo

```bash
python test_lstm_model.py
```

This will:
- Load the trained model
- Test on 5 sample EEG files
- Display predictions and accuracy metrics

### 2. Using the Model Programmatically

#### Load the Model

```python
import torch
from src.seq2seq_model import Seq2Seq
from src.vocabulary import Vocabulary

# Load checkpoint
checkpoint = torch.load('checkpoints/seq2seq_medium_model.pth',
                       map_location='cpu', weights_only=False)

# Get vocabulary and parameters
vocabulary = checkpoint['vocabulary']
downsample_factor = checkpoint.get('downsample_factor', 2)

# Initialize model
device = torch.device('cpu')
sequence_length = 5500 // downsample_factor  # 2750

model = Seq2Seq(
    vocab_size=len(vocabulary.word2idx),
    input_channels=105,
    sequence_length=sequence_length,
    embedding_dim=256,
    encoder_hidden_size=256,
    decoder_hidden_size=256,
    num_layers=2,
    dropout=0.3
).to(device)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
```

#### Predict from EEG Data

```python
import numpy as np

# Load your EEG data (105 channels × 5500 timesteps)
# eeg_data = ... (load from CSV or other source)

# Downsample
eeg_downsampled = eeg_data[:, ::downsample_factor]

# Prepare input
eeg_tensor = torch.FloatTensor(eeg_downsampled).unsqueeze(0).to(device)

# Generate prediction
with torch.no_grad():
    predicted_ids, attention_weights = model.generate(eeg_tensor, max_len=50)
    predicted_text = vocabulary.decode(predicted_ids[0].cpu().tolist())

print(f"Predicted: {predicted_text}")
```

#### Complete Example

```python
from src.data_loader import DataLoader

# Load data
data_loader = DataLoader('processed_data')
data_loader.load_mapping()

# Load a specific file
eeg_data = data_loader.load_padded_data('rawdata_6127.csv', target_length=5500)
true_text = data_loader.get_text_for_file('rawdata_6127.csv')

# Downsample and predict
eeg_downsampled = eeg_data[:, ::downsample_factor]
eeg_tensor = torch.FloatTensor(eeg_downsampled).unsqueeze(0).to(device)

with torch.no_grad():
    predicted_ids, _ = model.generate(eeg_tensor, max_len=50)
    predicted_text = vocabulary.decode(predicted_ids[0].cpu().tolist())

print(f"True:      {true_text}")
print(f"Predicted: {predicted_text}")
```

## Model Architecture

### Encoder-Decoder with Attention

```
Input EEG (105, 5500)
    ↓
Downsample 2x → (105, 2750)
    ↓
┌─────────────────────────────────────┐
│          ENCODER (LSTM)             │
│  - Bidirectional LSTM (2 layers)    │
│  - Hidden size: 256                 │
│  - Channel reduction: Yes           │
│  - Output: (batch, seq_len, 512)    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│      ATTENTION (Bahdanau)           │
│  - Aligns decoder with encoder      │
│  - Learns which parts of EEG to     │
│    focus on for each word           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│          DECODER (LSTM)             │
│  - Unidirectional LSTM (2 layers)   │
│  - Hidden size: 256                 │
│  - Word embedding: 256              │
│  - Generates words one-by-one       │
└─────────────────────────────────────┘
    ↓
Output Sentence (word sequence)
```

### Key Features

1. **Bidirectional Encoder**: Processes EEG signals in both directions for better context
2. **Attention Mechanism**: Learns to focus on relevant EEG features for each word
3. **Word-by-Word Generation**: Generates sentences sequentially with auto-regressive decoding
4. **Memory Optimized**: 2x EEG downsampling reduces memory by 50%

## Configuration

Key parameters in `src/config_lstm.py`:

```python
# Data
SEQUENCE_LENGTH = 5500           # Input EEG length
INPUT_CHANNELS = 105             # EEG channels

# Model
LSTM_HIDDEN_SIZE = 256          # Hidden units
LSTM_NUM_LAYERS = 2             # LSTM layers
LSTM_BIDIRECTIONAL = True       # Bidirectional encoder
EMBEDDING_DIM = 256             # Word embedding size

# Training (for reference)
DOWNSAMPLE_FACTOR = 2           # EEG downsampling
BATCH_SIZE = 4                  # Physical batch size
GRADIENT_ACCUMULATION = 4       # Effective batch: 16
```

## Trained Sentences

This model was trained on 5 specific sentences:

1. "However, the U.S. Navy accepted him in September of that year."
2. "In 1964 she went to Reprise again, shifting the next year to Dot Records."
3. "During this period, he published Profiles in Courage, highlighting eight instances..."
4. "He was reelected twice, but had a mixed voting record, often diverging from..."
5. "He is of three quarters Irish and one quarter French descent."

**Note**: The model performs best on EEG data from these sentences. For new sentences, retraining is recommended.

## Performance Metrics

### Validation Results (During Training)
- **Sentence Accuracy**: 40%
- **Word Error Rate**: 66%
- **Training Loss**: 0.26
- **Validation Loss**: 4.01

### Test Results (Current Model)
- **Word-Level Accuracy**: 61.4%
- **Average WER**: 38.6%
- **Perfect Matches**: 3/5 samples (60%)

**Note**: Word accuracy is higher than sentence accuracy because many predictions match perfectly except for capitalization/punctuation.

## Vocabulary

The model uses a vocabulary of **72 words** including:

**Special Tokens**:
- `<PAD>`: Padding token
- `<SOS>`: Start of sequence
- `<EOS>`: End of sequence
- `<UNK>`: Unknown word

**Most Common Words**: in, the, he, of, and, to, year, u.s, their, by

## Data Format

### Input EEG Data
- **Shape**: (105 channels, 5500 timesteps)
- **Format**: CSV file with 105 rows × variable columns
- **Preprocessing**: Zero-padding/truncation to 5500 timesteps

### Example Data Structure
```
processed_data/
├── sentence_mapping.csv      # Maps files to text
├── rawdata_0001.csv          # EEG data file
├── rawdata_0002.csv
└── ...
```

## Transferring to Another Project

### Minimum Files Needed

For inference only:
```
your_project/
├── checkpoints/
│   └── seq2seq_medium_model.pth
└── src/
    ├── seq2seq_model.py
    └── vocabulary.py
```

### Full Package

Copy the entire `lstm_model/` folder for complete functionality including data loading and testing.

## Troubleshooting

**Issue**: "Model file not found"
- **Solution**: Ensure `seq2seq_medium_model.pth` is in `checkpoints/` folder

**Issue**: "Vocabulary error" or "Module not found"
- **Solution**: Use `weights_only=False` when loading:
  ```python
  checkpoint = torch.load(path, map_location='cpu', weights_only=False)
  ```

**Issue**: "Low accuracy on test data"
- **Solution**: Ensure you're testing on the 5 sentences the model was trained on (see "Trained Sentences" section)

**Issue**: "Shape mismatch error"
- **Solution**: Verify EEG data is shape (105, 5500) and apply downsampling before inference

**Issue**: "Sentence marked WRONG but words match"
- **Solution**: This is due to case/punctuation differences. The model is working correctly - compare `word_accuracy` instead of sentence match.

## Advanced Usage

### Batch Prediction

```python
# Load multiple files
eeg_batch = []
for filename in ['rawdata_6127.csv', 'rawdata_6254.csv']:
    eeg = data_loader.load_padded_data(filename, target_length=5500)
    eeg_downsampled = eeg[:, ::downsample_factor]
    eeg_batch.append(eeg_downsampled)

# Stack into batch
eeg_tensor = torch.FloatTensor(np.array(eeg_batch)).to(device)

# Generate predictions
with torch.no_grad():
    predicted_ids, _ = model.generate(eeg_tensor, max_len=50)
    for i in range(len(eeg_batch)):
        text = vocabulary.decode(predicted_ids[i].cpu().tolist())
        print(f"Prediction {i+1}: {text}")
```

### Extracting Attention Weights

```python
# Get attention weights during generation
predicted_ids, attention_weights = model.generate(eeg_tensor, max_len=50)

# attention_weights is a list of tensors
# Each tensor: (batch_size, encoder_seq_len)
# Shows which EEG timesteps were focused on for each word

import matplotlib.pyplot as plt

# Visualize attention for first sample
attn = torch.stack(attention_weights).squeeze().cpu().numpy()
plt.imshow(attn, aspect='auto', cmap='viridis')
plt.xlabel('EEG Timestep')
plt.ylabel('Word Position')
plt.title('Attention Heatmap')
plt.colorbar()
plt.show()
```

## Citation

If you use this model in your research, please cite:

```
EEG-to-Text Seq2Seq LSTM Model
Trained on ZuCo dataset subset
5 sentences, 175 samples
40% validation accuracy, 61% word-level accuracy
```

## License

This package is part of an EEG-to-text ML project for educational purposes.
