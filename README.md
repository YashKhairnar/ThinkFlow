# Brain-to-Text: EEG Decoding Project

This project implements a Brain-to-Text system that decodes EEG signals into natural language text using a VQ-VAE and BART architecture. It consists of a Python Flask backend for inference and a Next.js frontend for demonstration.

## Project Structure

- **frontend/**: Next.js application for the demo interface.
- **inference.py**: Flask backend server that handles EEG data processing and text generation.
- **weights/**: Directory containing the trained model weights (`epoch_2.pt`).
- **test_data/**: Directory containing sample EEG CSV files.
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
3.  Select one of the pre-loaded test cases.
4.  (Optional) Click **View Tensor** to inspect the raw EEG data.
5.  The system will process the EEG data and display the decoded text.

## Model Architecture

The system employs a sophisticated two-stage architecture to translate raw EEG signals into natural language:

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

## References

If you use this work or find it helpful, please cite the original DeWave paper:

> Duan, Y., Zhou, J., Wang, Z., Wang, Y.-K., & Lin, C.-T. (2024). DeWave: Discrete EEG Waves Encoding for Brain Dynamics to Text Translation. arXiv preprint arXiv:2309.14030. [https://arxiv.org/abs/2309.14030](https://arxiv.org/abs/2309.14030)
