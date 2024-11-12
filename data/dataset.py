import matplotlib.pyplot as plt
import torchtext
import torch
import torch.nn as nn
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from model.model import tokenize_example, numericalize_example

class Dataset():
    def __init__(self) -> None:
        self.tokenizer = get_tokenizer("basic_english")
        pass

    def get_collate_fn(self, pad_index):
        def collate_fn(batch):
            batch_ids = [i["ids"] for i in batch]
            batch_ids = nn.utils.rnn.pad_sequence(
                batch_ids, padding_value=pad_index, batch_first=True
            )
            batch_label = [i["label"] for i in batch]
            batch_label = torch.stack(batch_label)
            batch = {"ids": batch_ids, "label": batch_label}
            return batch

        return collate_fn

    def get_data_loader(self, dataset, batch_size, pad_index, shuffle=False):
        collate_fn = self.get_collate_fn(pad_index)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
        )
        return data_loader

    def prepare_data(self, train_data, test_data, batch_size):
        max_length = 256
        train_data = train_data.map(
            tokenize_example, fn_kwargs={"tokenizer": self.tokenizer, "max_length": max_length}
        )
        test_data = test_data.map(
            tokenize_example, fn_kwargs={"tokenizer": self.tokenizer, "max_length": max_length}
        )

        test_size = 0.25

        train_valid_data = train_data.train_test_split(test_size=test_size)
        train_data = train_valid_data["train"]
        valid_data = train_valid_data["test"]

        min_freq = 5
        special_tokens = ["<unk>", "<pad>"]

        vocab = build_vocab_from_iterator(
            train_data["tokens"],
            min_freq=min_freq,
            specials=special_tokens,
        )
    def prepare_data(self, train_data, test_data, batch_size):
        """
        Prepare the training and testing data for PyTorch training.

        Args:
            train_data: The training dataset.
            test_data: The testing dataset.
            batch_size: The number of samples in each mini-batch.

        Returns:
            A tuple containing three data loaders (training, validation, and testing) and a vocabulary object.
        """
        max_length = 256
        train_data = train_data.map(
            tokenize_example, fn_kwargs={"tokenizer": self.tokenizer, "max_length": max_length}
        )
        test_data = test_data.map(
            tokenize_example, fn_kwargs={"tokenizer": self.tokenizer, "max_length": max_length}
        )

        test_size = 0.25

        train_valid_data = train_data.train_test_split(test_size=test_size)
        train_data = train_valid_data["train"]
        valid_data = train_valid_data["test"]

        min_freq = 5
        special_tokens = ["<unk>", "<pad>"]

        vocab = build_vocab_from_iterator(
            train_data["tokens"],
            min_freq=min_freq,
            specials=special_tokens,
        )
        
        unk_index = vocab["<unk>"]
        pad_index = vocab["<pad>"]
        vocab.set_default_index(unk_index)

        train_data = train_data.map(numericalize_example, fn_kwargs={"vocab": vocab})
        valid_data = valid_data.map(numericalize_example, fn_kwargs={"vocab": vocab})
        test_data = test_data.map(numericalize_example, fn_kwargs={"vocab": vocab})
    
        train_data = train_data.with_format(type="torch", columns=["ids", "label"])
        valid_data = valid_data.with_format(type="torch", columns=["ids", "label"])
        test_data = test_data.with_format(type="torch", columns=["ids", "label"])

        train_data_loader = self.get_data_loader(train_data, batch_size, pad_index, shuffle=True)
        valid_data_loader = self.get_data_loader(valid_data, batch_size, pad_index)
        test_data_loader = self.get_data_loader(test_data, batch_size, pad_index)

        return train_data_loader, valid_data_loader, test_data_loader, vocab, pad_index
