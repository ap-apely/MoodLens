from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
from torchtext.data import get_tokenizer
import torchtext
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


import uvicorn

import hydra
from omegaconf import OmegaConf

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import CNN

torchtext.disable_torchtext_deprecation_warning()

app = FastAPI(
    title="MoodLens API",
    description="API for analyze text by CNN",
    version="1.0.0"
)

class SentimentRequest(BaseModel):
    text: str

def get_config(cfg):
    config_data = torch.load(cfg.model.vocab_path)

    print(OmegaConf.to_yaml(cfg))
    
    # Доступ к параметрам модели
    embedding_dim = cfg.model.embedding_dim
    n_filters = cfg.model.n_filters
    filter_sizes = cfg.model.filter_sizes
    dropout_rate = cfg.model.dropout_rate
    vocab = config_data[0]
    output_dim = config_data[1]

    return embedding_dim, n_filters, filter_sizes, dropout_rate, vocab, output_dim

def init_model(cfg):
    global model
    global tokenizer
    global vocab
    global device

    embedding_dim, n_filters, filter_sizes, dropout_rate, vocab, output_dim = get_config(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = get_tokenizer("basic_english")

    model = CNN(vocab_size=len(vocab), embedding_dim=embedding_dim, n_filters=n_filters, filter_sizes=filter_sizes, output_dim=output_dim,dropout_rate=dropout_rate, pad_index=vocab["<pad>"]).to(device)
    model.load_state_dict(torch.load("../model/weights/cnn.pt"))
    model.eval()

def predict_sentiment(text, model, tokenizer, vocab, device, min_length=100, pad_index=vocab["<pad>"]):
    """
    Make a prediction on the sentiment of the given text.

    Args:
        text (str): The input text to be analyzed.
        model: The loaded CNN model.
        tokenizer: A tokenizer to split the input text into tokens.
        vocab: The vocabulary of the model.
        device: The device where the model is stored.
        min_length (int, optional): Minimum length of the input text. Defaults to 100.
        pad_index (int, optional): Index of the padding token in the vocabulary. Defaults to the index of the "<pad>" token.

    Returns:
        tuple: A tuple containing the predicted sentiment class and its corresponding probability.
    """
    tokens = tokenizer(text)
    ids = vocab.lookup_indices(tokens)
    if len(ids) < min_length:
        ids += [pad_index] * (min_length - len(ids))
    tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    prediction = model(tensor).squeeze(dim=0)
    probability = torch.softmax(prediction, dim=-1)
    predicted_class = prediction.argmax(dim=-1).item()
    predicted_probability = probability[predicted_class].item()
    return predicted_class, predicted_probability

@app.post("/predict/")
async def predict(request: SentimentRequest):
    text = request.text
    min_length = 50 
    pad_index = 0 
    predicted_class, predicted_probability = predict_sentiment(text, model, tokenizer, vocab, device, min_length, pad_index)
    
    sentiment = "Positive" if predicted_class == 1 else "Negative"
    return {"sentiment": sentiment, "probability": predicted_probability}

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", summary="Главная страница")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@hydra.main(config_path="../config/", config_name="config", version_base=None)
def startup(cfg):
    init_model(cfg=cfg)

if __name__ == "__main__":
    startup()
    uvicorn.run(app)