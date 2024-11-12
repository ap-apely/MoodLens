# 🌈 MoodLens: Sentiment Analysis Dashboard

MoodLens is a sentiment analysis dashboard designed to classify text data sentiment using a Convolutional Neural Network (CNN). Built on PyTorch and FastAPI, this dashboard provides a user-friendly web interface for sentiment predictions, focusing on IMDB movie review sentiment (positive or negative).

## 📚 Table of Contents
- [Overview](#overview)
- [🗂️ Project Structure](#project-structure)
- [⚙️ Setup](#setup)
- [🚀 Usage](#usage)
- [🧠 Model Architecture](#model-architecture)
- [⚙️ Configuration](#configuration)
- [📬 API Endpoints](#api-endpoints)
- [🛠️ Technologies Used](#technologies-used)
- [📜 Acknowledgments](#acknowledgments)

## Overview

MoodLens uses a CNN model trained on the IMDB dataset to analyze the sentiment of text data, specifically classifying it as positive or negative. It features:
- 🌐 **FastAPI** RESTful API for easy and fast predictions.
- 📊 **Interactive Dashboard** built with HTML templates.
- 🔥 **Pretrained Word Embeddings** with GloVe vectors for enhanced NLP.

## 🗂️ Project Structure

```plaintext
├── config                 # Configuration files (Hydra & OmegaConf)
├── dashboard
│   ├── static             # Static assets (CSS, JavaScript)
│   └── templates          # HTML templates for the dashboard
├── data
└── model
    ├── weights            # Model weights and saved configurations
```

## ⚙️ Setup

### Prerequisites

- Python 3.8 or later
- `pip` package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/MoodLens.git
   cd MoodLens
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download pretrained embeddings (if applicable):**
   Ensure GloVe embeddings are available in `.vector_cache`.

## 🚀 Usage

### Training the Model

To train the CNN model:
```bash
python train.py
```

This command trains a CNN-based sentiment classifier and saves the model weights to `model/weights/`.

### Running the Dashboard

To launch the API and web dashboard:
```bash
uvicorn app:app --reload
```

Access the dashboard at `http://127.0.0.1:8000` for sentiment analysis.

### Prediction Example

Send a POST request to `/predict/`:
```json
POST /predict/
{
    "text": "This movie was amazing!"
}
```

## 🧠 Model Architecture

MoodLens uses a custom CNN model architecture optimized for sentiment analysis:

```python
class CNN(nn.Module):
    def __init__(
        vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout_rate, pad_index
    ):
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        
        # Multiple 1D Convolutional layers for n-gram feature extraction
        self.convs = nn.ModuleList(
            [nn.Conv1d(embedding_dim, n_filters, filter_size) for filter_size in filter_sizes]
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)
```

### Model Explanation
- **Embedding Layer**: Transforms words into dense vector representations using pretrained embeddings (GloVe).
- **Convolutional Layers**: Each layer applies filters to extract meaningful n-grams from the embedded text. The `filter_sizes` parameter determines the n-gram sizes.
- **Pooling & Concatenation**: Extracts features from each filter and combines them for classification.
- **Fully Connected Layer**: Maps the concatenated features to the output classes (positive or negative).
  
The model is trained using cross-entropy loss and optimized with Adam. 

## ⚙️ Configuration

Configuration is managed through Hydra and OmegaConf, allowing you to modify parameters without changing code directly. See `config/config.yaml` for details on parameters like `embedding_dim`, `batch_size`, and `n_epochs`.

## 📬 API Endpoints

- `POST /predict/` - Accepts a JSON payload with a text string, returning the sentiment and probability.

## 🛠️ Technologies Used

- **Python**: PyTorch, FastAPI, torchtext, Hydra, OmegaConf, Rich library for console display.
- **HTML, CSS, JavaScript**: Frontend dashboard interface.
- **GloVe embeddings**: Pretrained word vectors.

## 📜 Acknowledgments

Big thanks to the creators of [FastAPI](https://fastapi.tiangolo.com/), [Hydra](https://hydra.cc/), [PyTorch](https://pytorch.org/), and [Rich](https://rich.readthedocs.io/). Their tools were instrumental in building MoodLens.