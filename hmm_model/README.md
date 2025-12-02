# EEG-to-Text HMM Model Package

This is a self-contained package for using pre-trained CNN and HMM models for EEG-to-text prediction.

## Contents

```
hmm_model/
├── checkpoints/
│   ├── cnn_encoder.pth      # Trained CNN encoder (673 KB)
│   └── hmm_models.pkl       # Trained HMM models (12 KB)
├── src/
│   ├── config.py            # Configuration parameters
│   ├── data_loader.py       # Data loading utilities
│   ├── feature_extractor.py # CNN encoder architecture
│   ├── hmm_model.py         # Gaussian HMM implementation
│   ├── predictor.py         # HMM predictor for sentences
│   └── utils.py             # Utility functions
├── test_single_file.py      # Test script for individual files
├── README.md                # This file
└── requirements.txt         # Python dependencies
```

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

### 1. Testing with Individual Files

If you have the `processed_data` folder with EEG data:

```bash
# Test a single file
python test_single_file.py --file rawdata_1234.csv

# Test multiple files
python test_single_file.py --file rawdata_1234.csv rawdata_5678.csv

# Show detailed word-level analysis
python test_single_file.py --file rawdata_1234.csv --verbose
```

### 2. Using Models Programmatically

#### Load and Use CNN Encoder

```python
import torch
import numpy as np
from src.feature_extractor import SupervisedCNNEncoder
from src.config import *

# Load CNN model
encoder = SupervisedCNNEncoder(
    input_channels=CNN_INPUT_CHANNELS,      # 105
    hidden_channels=CNN_HIDDEN_CHANNELS,    # 32
    num_classes=NUM_CLASSES,                # 344
    sequence_length=SEQUENCE_LENGTH         # 5500
)

checkpoint = torch.load('checkpoints/cnn_encoder.pth', map_location='cpu')
encoder.load_state_dict(checkpoint['model_state_dict'])
encoder.eval()

# Extract features from EEG data
# Input shape: (batch_size, 105, 5500)
eeg_data = torch.randn(1, 105, 5500)  # Example data
with torch.no_grad():
    features = encoder.get_features(eeg_data)
    # Output shape: (batch_size, 32, ~688)

print(f"Extracted features shape: {features.shape}")
```

#### Load and Use HMM Predictor

```python
from src.predictor import SentencePredictor
from src.config import *
import numpy as np

# Load HMM models
predictor = SentencePredictor(
    n_states=HMM_N_STATES,      # 5
    n_features=HMM_N_FEATURES   # 32
)
predictor.load('checkpoints/hmm_models.pkl')

# Predict sentence from features
# features_np shape: (time_steps, 32)
features_np = np.random.randn(688, 32)  # Example features

predicted_text, confidence = predictor.predict(features_np)
print(f"Predicted: {predicted_text}")
print(f"Confidence: {confidence}")

# Get top 3 predictions
top_3 = predictor.predict(features_np, top_k=3)
for i, (text, score) in enumerate(top_3, 1):
    print(f"{i}. {text[:50]}... (score: {score:.2f})")
```

#### Complete Pipeline

```python
import torch
import numpy as np
from src.feature_extractor import SupervisedCNNEncoder
from src.predictor import SentencePredictor
from src.data_loader import DataLoader
from src.config import *

# 1. Load models
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

# 2. Load your EEG data (shape: 105 channels × time_steps)
# Example: Load from CSV file
loader = DataLoader('processed_data')
loader.load_mapping()
eeg_data = loader.load_padded_data('rawdata_1234.csv', target_length=SEQUENCE_LENGTH)

# 3. Extract features with CNN
X = torch.tensor(np.array([eeg_data]), dtype=torch.float32)
with torch.no_grad():
    features = encoder.get_features(X)
    features_np = features.cpu().numpy()

# 4. Normalize features (simple standardization)
features_normalized = (features_np[0].T - features_np[0].T.mean(axis=0)) / (features_np[0].T.std(axis=0) + 1e-8)

# 5. Predict sentence
predicted_text, confidence = predictor.predict(features_normalized)
print(f"Prediction: {predicted_text}")
print(f"Confidence: {confidence:.2f}")
```

## Model Architecture

### CNN Encoder (SupervisedCNNEncoder)
- **Input**: (batch, 105 channels, 5500 time steps)
- **Architecture**: 3 convolutional layers with BatchNorm and Dropout
- **Output**: (batch, 32 features, ~688 time steps)
- **Training**: Supervised classification on 344 sentence classes

### HMM Models (GaussianHMM)
- **States**: 5 hidden states per sentence
- **Features**: 32-dimensional feature vectors
- **Covariance**: Diagonal (for stability)
- **Training**: Baum-Welch algorithm (10 iterations)
- **Number of models**: One HMM per unique sentence (344 total)

## Configuration

Key parameters in `src/config.py`:

```python
# CNN Parameters
CNN_INPUT_CHANNELS = 105      # EEG channels
CNN_HIDDEN_CHANNELS = 32      # Feature dimension
SEQUENCE_LENGTH = 5500        # Input time steps

# HMM Parameters
HMM_N_STATES = 5              # Hidden states per sentence
HMM_N_FEATURES = 32           # Must match CNN output
NUM_CLASSES = 344             # Number of unique sentences
```

## Notes

1. **Input Data Format**: EEG data should be shape (105, T) where:
   - 105 = number of channels
   - T = time steps (will be padded/truncated to 5500)

2. **Feature Normalization**: The test script uses a simple standardization scaler. For better results, save the scaler used during training and load it here.

3. **Model Compatibility**: Ensure CNN and HMM models are from the same training run (same number of classes).

4. **GPU Usage**: Models run on CPU by default. To use GPU, modify the device in your code:
   ```python
   encoder = encoder.to('cuda')
   eeg_data = eeg_data.to('cuda')
   ```

## Transferring to Another Project

To use these models in a different project:

1. **Minimum files needed**:
   - `checkpoints/` folder (both .pth and .pkl files)
   - `src/feature_extractor.py` (for CNN)
   - `src/hmm_model.py` (for HMM)
   - `src/predictor.py` (for HMM inference)

2. **For full functionality**, copy the entire `hmm_model/` folder

3. **Update paths** in your code if you reorganize the structure

## Troubleshooting

**Issue**: "Class mismatch between CNN and HMM"
- **Solution**: Ensure both models are from the same training run

**Issue**: "Invalid data shape"
- **Solution**: Check that input is (105, 5500). Use `DataLoader.load_padded_data()` to ensure correct shape

**Issue**: "Feature normalization errors"
- **Solution**: Ensure features are normalized before passing to HMM predictor

## License

This package is part of an EEG-to-text ML project.
