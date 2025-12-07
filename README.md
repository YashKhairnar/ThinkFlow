# Brain-to-Text: EEG Decoding Project

This project implements a Brain-to-Text system that decodes EEG signals into natural language text using two different deep learning architectures: a **Transformer-based model (VQ-VAE + BART)** and a **Sequence-to-Sequence LSTM with Attention**. It consists of a Python Flask backend for inference and a Next.js frontend for interactive demonstration and model comparison.

## Project Structure

- **frontend/**: Next.js application for the demo interface with side-by-side model comparison.
- **inference.py**: Flask backend server that handles EEG data processing and text generation for both models.
- **weights/**: Directory containing the trained Transformer model weights (`epoch_49.pt`).
- **lstm_model/**: LSTM model implementation and resources.
  - **src/**: Source code for LSTM architecture (encoder, decoder, attention, vocabulary).
  - **checkpoints/**: Trained LSTM model weights (`seq2seq_medium_model.pth`).
  - **processed_data/**: Sample EEG data for LSTM testing.
  - **test_lstm_model.py**: Standalone script to test the LSTM model.
- **test_data/**: Directory containing sample EEG CSV files for Transformer model.
- **requirements.txt**: Python dependencies.

## Prerequisites

- Python 3.9+
- Node.js 18+
- npm or yarn

## Setup

### 1. Backend Setup

It is recommended to use a virtual environment.

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Frontend Setup

```bash
cd frontend
npm install
```

## Running the Application

### 1. Start the Backend Server

Ensure your virtual environment is activated.

```bash
# From the project root
python3 inference.py
```

The server will start on `http://127.0.0.1:5000`.

### 2. Start the Frontend Application

Open a new terminal window.

```bash
cd frontend
npm run dev
```

The application will be available at `http://localhost:3000`.

## Usage

1.  Open `http://localhost:3000` in your browser.
2.  Navigate to the **Demo** page.
3.  You will see two model sections side-by-side:
    - **Transformer Model** (left): VQ-VAE + BART architecture
    - **LSTM Model** (right): Sequence-to-Sequence with Attention
4.  Select test cases from either model to compare their performance.
5.  (Optional) Click **View Tensor** to inspect the raw EEG data.
6.  Click **Decode to Text** to process the EEG signal.
7.  The system will display:
    - Generated output from the model
    - Expected ground truth text
    - Confidence score (for LSTM model)

## Model Architectures

This project implements two different approaches for EEG-to-text translation, allowing for performance comparison and architectural insights.

---

## A. Transformer-Based Model (VQ-VAE + BART)

The Transformer model employs a sophisticated two-stage architecture to translate raw EEG signals into natural language:

### 1. EEG Encoder (VQ-VAE)

*   **Convolutional Layers**: A series of 1D convolutional layers extract local temporal features from the raw EEG data (105 channels → 64 → 128 → 256 → 512 channels).
*   **Multi-Head Attention**: A custom Transformer-based attention mechanism (8 heads) captures long-range dependencies across the time series. This allows the model to focus on relevant parts of the EEG signal regardless of their temporal distance.
*   **Vector Quantization (VQ) with Codebook**: 
    - The continuous feature vectors are mapped to a **discrete codebook (codex)** of 2048 learned embeddings (512-dimensional each).
    - During training, the model learns both the encoder and this codebook simultaneously using two loss terms:
        - **Codebook Loss**: Ensures the codebook embeddings move closer to the encoder outputs.
        - **Commitment Loss**: Encourages the encoder to commit to codebook entries.
    - This effectively tokenizes brain activity into a sequence of 57 discrete units, creating a "brain vocabulary."

### 2. Text Generation (BART)

*   **Pre-trained Transformer**: We utilize **BART (Bidirectional and Auto-Regressive Transformers)**, a powerful sequence-to-sequence model pre-trained by Facebook.
*   **Projection Layer**: A linear layer projects the 512-dimensional EEG tokens to BART's 1024-dimensional embedding space.
*   **Cross-Modal Translation**: The BART decoder attends to these "brain tokens" to generate coherent English sentences, leveraging its extensive knowledge of language structure.

### Training Process

The model is trained end-to-end with:
- **VQ Loss**: Codebook + commitment losses for learning discrete representations
- **Reconstruction Loss**: Ensures the generated text matches the ground truth
- The codebook learns to capture meaningful patterns in brain activity that correspond to linguistic concepts

**Model Specifications:**
- **Input**: EEG signals (105 channels, 5500 timesteps)
- **Encoder Output**: 57 discrete tokens (from 2048-entry codebook)
- **Final Output**: Decoded natural language text

---

## B. LSTM Sequence-to-Sequence Model with Attention

The LSTM model implements a classical encoder-decoder architecture enhanced with Bahdanau attention mechanism:

### 1. EEG Encoder (Bidirectional LSTM)

*   **Channel Reduction (Optional)**: A convolutional preprocessing stage reduces input dimensionality:
    - **Conv1D Layer 1**: 105 channels → 64 channels (kernel_size=3, BatchNorm, ReLU, Dropout=0.2)
    - **Conv1D Layer 2**: 64 channels → 32 channels (kernel_size=3, BatchNorm, ReLU)
    - This significantly reduces computational cost while preserving signal information.
*   **Bidirectional LSTM**:
    - **Architecture**: 2-layer bidirectional LSTM with hidden size of 256
    - **Input**: Either raw 105-channel or reduced 32-channel EEG data (5500 timesteps, downsampled to 2750)
    - **Output**: Contextual representations from both forward and backward passes (hidden_size × 2 = 512 dimensions)
    - **Dropout**: 0.3 between LSTM layers for regularization

### 2. Bahdanau Attention Mechanism

*   **Additive Attention**: Computes dynamic alignment between decoder state and encoder outputs
*   **Components**:
    - **W_encoder**: Linear transformation of encoder outputs (512 → 256)
    - **W_decoder**: Linear transformation of decoder hidden state (256 → 256)
    - **V**: Attention scoring layer (256 → 1)
*   **Context Vector**: Weighted sum of encoder outputs based on attention weights
*   **Benefit**: Allows the decoder to focus on relevant parts of the EEG signal at each decoding step

### 3. Text Generation (LSTM Decoder)

*   **Word Embeddings**: Learned embeddings (vocabulary size × 256 dimensions)
*   **LSTM Decoder**:
    - **Architecture**: 2-layer unidirectional LSTM (hidden size=256)
    - **Input**: Concatenation of word embedding and attention context vector
    - **Output**: Hidden representations for word prediction
*   **Prediction Head**:
    - Combines LSTM output, context vector, and word embedding
    - Two-layer feedforward network with ReLU activation
    - Final layer projects to vocabulary size for word probabilities

### Training Process

The LSTM model is trained with:
- **Cross-Entropy Loss**: Standard sequence-to-sequence training objective
- **Teacher Forcing**: Uses ground truth tokens as decoder input with 50% probability during training
- **Optimization**: Adam optimizer with learning rate scheduling
- **Data Augmentation**: Downsampling (factor of 2) to reduce sequence length and improve training efficiency

**Model Specifications:**
- **Input**: EEG signals (105 channels, 5500 timesteps)
- **Downsampling**: 2× reduction → 2750 timesteps
- **Channel Reduction**: 105 → 32 channels (optional)
- **Encoder Output**: Sequence of 512-dimensional contextualized vectors
- **Decoder**: Generates words autoregressively with attention
- **Parameters**: ~15M trainable parameters
- **Final Output**: Decoded natural language text with confidence scores

### Vocabulary Management

*   **Custom Vocabulary Class**: Handles word-to-index and index-to-word mappings
*   **Special Tokens**: `<PAD>`, `<SOS>` (start-of-sequence), `<EOS>` (end-of-sequence), `<UNK>` (unknown)
*   **Vocabulary Size**: Dynamically built from training corpus

---

## Model Comparison

| Feature | Transformer (VQ-VAE + BART) | LSTM with Attention |
|---------|----------------------------|---------------------|
| **Architecture** | Two-stage: VQ-VAE encoder + BART decoder | Encoder-Decoder with attention |
| **Representation** | Discrete codebook (2048 entries) | Continuous hidden states |
| **Context Modeling** | Multi-head self-attention (8 heads) | Bidirectional LSTM |
| **Sequence Modeling** | Parallel (Transformer) | Sequential (LSTM) |
| **Pre-training** | Leverages pre-trained BART | Trained from scratch |
| **Parameters** | ~400M (BART-large) + custom encoder | ~15M |
| **Inference Speed** | Faster (parallel processing) | Slower (sequential) |
| **Interpretability** | Codebook analysis | Attention weights |

---

## References

If you use this work or find it helpful, please cite the original DeWave paper:

> Duan, Y., Zhou, J., Wang, Z., Wang, Y.-K., & Lin, C.-T. (2024). DeWave: Discrete EEG Waves Encoding for Brain Dynamics to Text Translation. arXiv preprint arXiv:2309.14030. [https://arxiv.org/abs/2309.14030](https://arxiv.org/abs/2309.14030)
