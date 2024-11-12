import datasets
import torchtext
import torch.nn as nn
import torch.optim as optim
import torch
import tqdm
import numpy as np
import collections
import matplotlib.pyplot as plt
import hydra
import omegaconf
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from rich import box

from data.dataset import Dataset
from model.model import CNN, initialize_weights

console = Console()

def count_parameters(model):
    """Return the total number of trainable parameters in a given PyTorch model.
    
    Args:
        model (nn.Module): The PyTorch model to count parameters for.
    
    Returns:
        int: The total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def predict_sentiment(text, model, tokenizer, vocab, device, min_length, pad_index):
    """Make a prediction about the sentiment of a given text based on a trained PyTorch model.
    
    Args:
        text (str): The input text to make a prediction for.
        model: A PyTorch neural network model that has been trained on a sentiment analysis task.
        tokenizer: A PyTorch Text Tokenizer object that can break down the input text into individual tokens.
        vocab: A PyTorch Vocabulary object that maps token IDs to their corresponding word embeddings.
        device (torch.device): The device (GPU or CPU) on which to run the model.
        min_length (int): The minimum length of a sequence before padding with zeros.
        pad_index (int): The index used for padding sequences shorter than the specified minimum length.
    
    Returns:
        tuple: A tuple containing the predicted sentiment class and its corresponding probability score.
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

def get_accuracy(prediction, label):
    """Calculate the accuracy of a model's predictions based on a given batch of true labels.
    
    Args:
        prediction (torch.Tensor): A tensor containing the model's predicted sentiment classes for a batch of input texts.
        label (torch.Tensor): A tensor containing the true sentiment labels for the same batch of input texts.
    
    Returns:
        float: The accuracy of the model's predictions as a decimal value between 0 and 1.
    """
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = (predicted_classes == label).sum().item()
    accuracy = correct_predictions / batch_size
    return accuracy

def train(data_loader, model, criterion, optimizer, device):
    """
    Trains the model on a given data loader for a specified number of epochs.
    
    Args:
        data_loader (torch.utils.data.DataLoader): The data loader containing batches of training data.
        model: A PyTorch neural network model that has been initialized and compiled.
        criterion: A loss function used to evaluate the model's performance on a given batch of input data.
        optimizer: An optimization algorithm used to update the model's weights during training.
        device (torch.device): The device (GPU or CPU) on which to run the model.
    
    Returns:
        tuple: A tuple containing the average loss and accuracy over all epochs.
    """
    model.train()
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm.tqdm(data_loader, desc="training..."):
        ids = batch["ids"].to(device)
        label = batch["label"].to(device)
        prediction = model(ids)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy)
    return np.mean(epoch_losses), np.mean(epoch_accs)

def evaluate(data_loader, model, criterion, device):
    """Evaluates the performance of a trained PyTorch model on a given data loader.
    
    Args:
        data_loader (torch.utils.data.DataLoader): The data loader containing batches of evaluation data.
        model: A PyTorch neural network model that has been initialized and compiled.
        criterion: A loss function used to evaluate the model's performance on a given batch of input data.
        device (torch.device): The device (GPU or CPU) on which to run the model.
    
    Returns:
        tuple: A tuple containing the average loss and accuracy over all batches in the data loader.
    """
    model.eval()
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="evaluating..."):
            ids = batch["ids"].to(device)
            label = batch["label"].to(device)
            prediction = model(ids)
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy)
    return np.mean(epoch_losses), np.mean(epoch_accs)

def create_graph(metrics, n_epochs):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(metrics["train_losses"], label="training loss")
    ax.plot(metrics["valid_losses"], label="validation loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_xticks(range(n_epochs))
    ax.legend()
    ax.grid(True)  # Add grid lines to the plot
    fig.suptitle("Model Training and Validation Loss Over Epochs")  # Add a title to the figure
    plt.show()  # Display the plot

@hydra.main(config_path="./config/", config_name="config", version_base=None)
def start_train(cfg):
    """
    This function trains a CNN model on the IMDB dataset using the train_data, valid_data, and test_data loaders.
    
    Returns:
        None
    """
    console.log("Initializing dataset and model...")
    dataset = Dataset()

    batch_size = cfg.training.batch_size
    train_data, test_data = datasets.load_dataset("imdb", split=["train", "test"])
    train_data_loader, valid_data_loader, test_data_loader, vocab, pad_index = dataset.prepare_data(train_data, test_data, batch_size)
    
    vocab_size = len(vocab)
    embedding_dim = cfg.model.embedding_dim
    n_filters = cfg.model.n_filters
    filter_sizes = cfg.model.filter_sizes
    dropout_rate = cfg.model.dropout_rate
    output_dim = len(train_data.unique("label"))

    model = CNN(
        vocab_size,
        embedding_dim,
        n_filters,
        filter_sizes,
        output_dim,
        dropout_rate,
        pad_index,
    )
    torch.save([vocab, output_dim], './model/weights/config.pth')
    console.log(f"[bold green]Model initialized with {count_parameters(model):,} trainable parameters[/bold green]")

    model.apply(initialize_weights)
    vectors = torchtext.vocab.GloVe()
    pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos())
    model.embedding.weight.data = pretrained_embedding

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)

    n_epochs = cfg.training.n_epochs
    best_valid_loss = float("inf")
    metrics = collections.defaultdict(list)

    # Table setup for displaying epoch results
    table = Table(title="Training Progress", box=box.SQUARE)
    table.add_column("Epoch", justify="center", style="cyan", no_wrap=True)
    table.add_column("Train Loss", justify="right", style="magenta")
    table.add_column("Train Acc", justify="right", style="magenta")
    table.add_column("Valid Loss", justify="right", style="yellow")
    table.add_column("Valid Acc", justify="right", style="yellow")

    with Progress(TextColumn("[progress.description]{task.description}"),
              BarColumn(),
              TaskProgressColumn()) as progress:
        epoch_task = progress.add_task("[green]Training...", total=n_epochs)

        for epoch in range(n_epochs):
            # Perform training and validation
            train_loss, train_acc = train(train_data_loader, model, criterion, optimizer, device)
            valid_loss, valid_acc = evaluate(valid_data_loader, model, criterion, device)
            
            # Save metrics and model as before
            metrics["train_losses"].append(train_loss)
            metrics["train_accs"].append(train_acc)
            metrics["valid_losses"].append(valid_loss)
            metrics["valid_accs"].append(valid_acc)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), "./model/weights/cnn.pt")
            
            # Recreate the table for each epoch
            table = Table(title="Training Progress", box=box.SQUARE)
            table.add_column("Epoch", justify="center", style="cyan", no_wrap=True)
            table.add_column("Train Loss", justify="right", style="magenta")
            table.add_column("Train Acc", justify="right", style="magenta")
            table.add_column("Valid Loss", justify="right", style="yellow")
            table.add_column("Valid Acc", justify="right", style="yellow")
            
            # Add current epoch data
            table.add_row(str(epoch + 1), f"{train_loss:.3f}", f"{train_acc:.3f}", f"{valid_loss:.3f}", f"{valid_acc:.3f}")
            
            # Print the table for this epoch
            console.print(table)
            
            # Update progress bar
            progress.update(epoch_task, advance=1)
            progress.refresh()
    
    console.log("[bold green]Training completed. Evaluating on test data...[/bold green]")

    model.load_state_dict(torch.load("cnn.pt"))
    test_loss, test_acc = evaluate(test_data_loader, model, criterion, device)
    console.print(f"[bold magenta]Test Loss:[/bold magenta] {test_loss:.3f}, [bold magenta]Test Accuracy:[/bold magenta] {test_acc:.3f}")

    create_graph(metrics=metrics, n_epochs=n_epochs)

if __name__ == "__main__":
    start_train()