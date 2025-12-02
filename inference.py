import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoTokenizer, AutoModel, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
import pandas as pd
import numpy as np
import flask
from flask import request, jsonify
from flask_cors import CORS
import sys


# LSTM model imports
sys.path.append('lstm_model')
sys.path.append('lstm_model/src')
from src.seq2seq_model import Seq2Seq
from src.vocabulary import Vocabulary
# Import vocabulary module to allow unpickling
import vocabulary



app = flask.Flask(__name__)
CORS(app)  # Enable CORS for all routes


import os

def load_and_pad_eeg(filepath: str) -> torch.Tensor:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"EEG file not found: {filepath}")

    # Expect shape [channels=105, timesteps]; adjust if your CSV is transposed
    df = pd.read_csv(filepath)
    arr = df.values.astype(np.float32)

    # If CSV is [T, 105] instead of [105, T], transpose:
    if arr.shape[0] == 5500 and arr.shape[1] == 105:
        arr = arr.T

    # Pad or crop to pad_len along time axis (axis=1)
    pad_len = 5500
    c, t = arr.shape
    if c != 105:
        raise ValueError(f"Expected 105 channels, got {c} in {filepath}")

    if t < pad_len:
        pad = np.zeros((c, pad_len - t), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    elif t > pad_len:
        arr = arr[:, :pad_len]

    return torch.from_numpy(arr).to(device)  # [105, pad_len]


class ConvolutionModel(nn.Module):
    '''
    Input : single sentence EEG raw input of (105 channels, 5500 timestamps)
    Output : single sentence of 57 features of 512 dimensions embedding each
    '''
    def __init__(self):
        super().__init__()
        self.convolutional_model = nn.Sequential(
            nn.Conv1d(in_channels=105, kernel_size=10, out_channels=64, stride=3),
            nn.Conv1d(in_channels=64,  kernel_size=3,  out_channels=128, stride=2),
            nn.Conv1d(in_channels=128, kernel_size=3,  out_channels=256, stride=2),
            nn.Conv1d(in_channels=256, kernel_size=3,  out_channels=512, stride=2),
            nn.Conv1d(in_channels=512, kernel_size=2,  out_channels=512, stride=2),
            nn.Conv1d(in_channels=512, kernel_size=2,  out_channels=512, stride=2),
        )

    def forward(self, x):
        # Input shape expected: [batch_size, channels, timestamps]
        op = self.convolutional_model(x)
        # Output shape is [batch_size, d_model, num_tokens] -> transpose to [batch_size, num_tokens, d_model]
        return op.permute(0, 2, 1)

class AttentionBlock(nn.Module):
    """
    Single-head scaled dot-product attention using nn.Linear layers.
    """
    def __init__(self, d_model, d_k):
        super().__init__()
        self.d_k = d_k
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_k, bias=False)

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        output = attn_weights @ V
        return output

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, heads=8):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        # d_k is the dimension of each head. Must be divisible by d_model
        self.d_k = d_model // heads

        # Use nn.ModuleList to register attention blocks with PyTorch
        self.attentionBlocks = nn.ModuleList([AttentionBlock(d_model, self.d_k) for _ in range(self.heads)])

        # Final linear layer to project concatenated heads back to d_model dimension
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        head_outputs = []
        for block in self.attentionBlocks:
            # Assuming x is [batch_size, num_tokens, d_model]
            out_h = block(x) # [batch_size, num_tokens, d_k]
            head_outputs.append(out_h)

        # Concatenate along the last dimension
        total = torch.cat(head_outputs, dim=-1) # [batch_size, num_tokens, d_model]

        # Apply final linear layer
        output = self.output_linear(total)
        return output

class Encoder(nn.Module):
    def __init__(self, d_model=512, heads=8, beta=0.2):
        super().__init__()
        self.conv = ConvolutionModel()
        self.mha = MultiHeadAttentionBlock(d_model=d_model, heads=heads)
        self.codex = nn.Embedding(num_embeddings=2048, embedding_dim=512)
        self.words = self.codex.weight  # codebook
        self.beta = beta               # commitment weight

    def forward(self, x):
        """
        x: [batch_size, 105, 5500]

        returns:
            z_q_st: [batch_size, 57, 512]  (quantized, straight-through)
            vq_loss: scalar (codebook + commitment terms)
            indices: [batch_size, 57]  (chosen code indices)
        """
        # conv_output: [batch_size, 57, 512]
        conv_output = self.conv(x)

        # attn_output: [batch_size, 57, 512] (this is z_c in VQ-VAE terms)
        attn_output = self.mha(conv_output)

        # --- vector quantization using your codex / words ---
        B, L, D = attn_output.shape  # [B, 57, 512]

        codebook = self.words.unsqueeze(0)
        #atthention output of size [ batch_size, 57, 512 ] but its' context aware now
        distances = torch.cdist(attn_output, codebook) # distances of each of 57 EEG feature with the 2048 words in codex book

        # indices of the least-distance codex word for each EEG feature
        # indices: [B, 57]
        indices = torch.argmin(distances, dim=-1)

        z_q = self.words[indices]

        # --- VQ codebook + commitment losses ---

        # codebook loss: || sg[attn_output] - z_q ||^2  (update codex/words)
        codebook_loss = F.mse_loss(z_q, attn_output.detach())

        # commitment loss: || attn_output - sg[z_q] ||^2  (update encoder)
        commitment_loss = F.mse_loss(attn_output, z_q.detach())

        vq_loss = codebook_loss + self.beta * commitment_loss

        # straight-through: forward uses z_q, grads go to attn_output
        z_q_st = attn_output + (z_q - attn_output).detach()

        return z_q_st, vq_loss, indices

def load_model(model_path: str, device: torch.device):
    """
    Loads a PyTorch model from a .pt file.
    Handles both:
      - entire model saved (torch.save(model, path))
      - checkpoint dict with 'encoder_state' key
      - direct state_dict saved (torch.save(model.state_dict(), path))
    """
    obj = torch.load(model_path, map_location=device)

    # Case 1: entire model
    if isinstance(obj, torch.nn.Module):
        model = obj
        model.to(device)
        return model

    # Case 2: checkpoint dictionary (with 'encoder_state', 'decoder_state', etc.)
    elif isinstance(obj, dict):
        # Check if this is a training checkpoint with encoder_state
        if 'encoder_state' in obj:
            model = Encoder() 
            model.load_state_dict(obj['encoder_state'])
            model.to(device)
            return model
        # Otherwise, assume it's a direct state_dict
        else:
            model = Encoder() 
            model.load_state_dict(obj)
            model.to(device)
            return model

    else:
        raise ValueError("Loaded object is neither a nn.Module nor a state_dict.")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    filename = data.get('filename')
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
        
    try:
        # Load and preprocess the EEG data from file
        input_eeg = load_and_pad_eeg(filename).unsqueeze(0) # Add batch dimension [1, 105, 5500]
        
        with torch.no_grad():
            # Get encoder output: [1, 57, 512]
            encoder_output, vq_loss, indices = encoder(input_eeg)
            
            # Project to BART's hidden dimension: [1, 57, 1024]
            bart_input = projection(encoder_output)
            
            # Generate text using BART decoder
            encoder_outputs = BaseModelOutput(
                last_hidden_state=bart_input
            )
            
            # Generate text using BART decoder
            generated_ids = bart_model.generate(
                encoder_outputs=encoder_outputs,
                max_length=50,
                num_beams=5,
                early_stopping=True
            )
            
            # Decode the generated tokens to text
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            return {'generated_text': generated_text}
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

# Initialize models globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model on {device}...")

# Load the encoder model
encoder = load_model("weights/epoch_2.pt", device)
encoder.eval()

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large").to(device)

# Create a projection layer to match BART's hidden size (1024)
projection = nn.Linear(512, 1024).to(device)
print("Model loaded successfully!")



# --- LSTM Model Initialization ---
print("Loading LSTM model...")
# Load LSTM checkpoint
lstm_checkpoint = torch.load('lstm_model/checkpoints/seq2seq_medium_model.pth', map_location=device, weights_only=False)

# Extract vocabulary from checkpoint
lstm_vocab = lstm_checkpoint['vocabulary']
vocab_size = len(lstm_vocab.word2idx)
downsample_factor = lstm_checkpoint.get('downsample_factor', 2)
sequence_length_downsampled = 5500 // downsample_factor

print(f"Vocabulary size: {vocab_size}")
print(f"Downsample factor: {downsample_factor}")

# Initialize LSTM model
lstm_model = Seq2Seq(
    vocab_size=vocab_size,
    input_channels=105,
    sequence_length=sequence_length_downsampled,
    embedding_dim=256,
    encoder_hidden_size=256,
    decoder_hidden_size=256,
    num_layers=2,
    dropout=0.3,
    use_channel_reduction=True,
    reduced_channels=32
)

# Load model weights
lstm_model.load_state_dict(lstm_checkpoint['model_state_dict'])
lstm_model.to(device)
lstm_model.eval()
print("LSTM Model loaded successfully!")



@app.route('/predict_lstm', methods=['POST'])
def predict_lstm():
    data = request.json
    filename = data.get('filename')
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
        
    try:
        # Load and preprocess the EEG data from file
        eeg_data = load_and_pad_eeg(filename) # [105, 5500]
        
        # Downsample the EEG data
        eeg_downsampled = eeg_data[:, ::downsample_factor]  # [105, 2750]
        input_eeg = eeg_downsampled.unsqueeze(0)  # [1, 105, 2750]
        
        with torch.no_grad():
            # Generate text using LSTM model
            generated_indices, attention_weights = lstm_model.generate(
                input_eeg,
                max_len=50,
                sos_idx=lstm_vocab.sos_idx,
                eos_idx=lstm_vocab.eos_idx
            )
            
            # Decode indices to text
            generated_text = lstm_vocab.decode(generated_indices[0].cpu().tolist())
            
            # Calculate confidence (average of max probabilities - approximation)
            confidence = 0.85  # Placeholder since we don't have direct probability output
            
            return {'generated_text': generated_text, 'confidence': float(confidence)}
            
    except Exception as e:
        print(f"Error during LSTM prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)