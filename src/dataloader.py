from pprint import pprint

import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizer

from config import config


class FeedBackDataset(Dataset):
    def __init__(self, df, tokenizer: PreTrainedTokenizer, max_length):
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.text = df["text"].values
        self.targets = df["discourse_effectiveness"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.text[index]
        if config['text_lowercase']:
            text = text.lower()
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=config["max_length"],
        )

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "target": self.targets[index],
        }


class Collate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        output["target"] = [sample["target"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [
                s + (batch_max - len(s)) * [self.tokenizer.pad_token_id]
                for s in output["input_ids"]
            ]
            output["attention_mask"] = [
                s + (batch_max - len(s)) * [0] for s in output["attention_mask"]
            ]
        else:
            output["input_ids"] = [
                (batch_max - len(s)) * [self.tokenizer.pad_token_id] + s
                for s in output["input_ids"]
            ]
            output["attention_mask"] = [
                (batch_max - len(s)) * [0] + s for s in output["attention_mask"]
            ]

        # convert to tensors
        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(
            output["attention_mask"], dtype=torch.long
        )
        output["target"] = torch.tensor(output["target"], dtype=torch.long)

        return output


def get_loaders(fold, accelerator):
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], use_fast=True)
    # collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

    df = pd.read_csv(config["train_folds"])
    df_train = df[df.fold != fold].reset_index(drop=True)
    df_valid = df[df.fold == fold].reset_index(drop=True)

    train_dataset = FeedBackDataset(
        df_train, tokenizer=tokenizer, max_length=config["max_length"]
    )
    valid_dataset = FeedBackDataset(
        df_valid, tokenizer=tokenizer, max_length=config["max_length"]
    )

    collate_fn = Collate(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train_bs"],
        collate_fn=collate_fn,
        num_workers=2,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["valid_bs"],
        collate_fn=collate_fn,
        num_workers=2,
        shuffle=False,
        pin_memory=False,
    )

    return train_loader, valid_loader


if __name__ == "__main__":
    df = pd.read_csv(config["train_folds"])
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], use_fast=True)

    train_dataset = FeedBackDataset(
        df, tokenizer=tokenizer, max_length=config["max_length"]
    )
    collate_fn = Collate(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train_bs"],
        collate_fn=collate_fn,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    for data in train_loader:
        print(data)
        print(data['input_ids'].size())
        print(data['attention_mask'].size())
        print(data['target'].size())
        break
