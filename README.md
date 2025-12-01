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

1.  **EEG Encoder (VQ-VAE)**:
    *   **Convolutional Layers**: A series of 1D convolutional layers extract local temporal features from the raw EEG data (105 channels).
    *   **Multi-Head Attention**: A custom Transformer-based attention mechanism captures long-range dependencies across the time series. This allows the model to focus on relevant parts of the EEG signal regardless of their temporal distance.
    *   **Vector Quantization (VQ)**: The continuous feature vectors are mapped to a discrete codebook (codex), effectively tokenizing the brain activity into a sequence of discrete units.

2.  **Text Generation (BART)**:
    *   **Pre-trained Transformer**: We utilize **BART (Bidirectional and Auto-Regressive Transformers)**, a powerful sequence-to-sequence model pre-trained by Facebook.
    *   **Cross-Modal Translation**: The discrete EEG tokens from the encoder are projected into the BART embedding space. The BART decoder then attends to these "brain tokens" to generate coherent English sentences, leveraging its extensive knowledge of language structure.

- **Input**: EEG signals (105 channels, 5500 timesteps).
- **Output**: Decoded natural language text.
