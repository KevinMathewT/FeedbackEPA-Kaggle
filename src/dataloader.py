import pandas as pd

from config import config

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding


class FeedBackDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.discourse_type = df["discourse_type"].values
        self.discourse = df["discourse_text"].values
        self.essay = df["text"].values
        self.targets = df["discourse_effectiveness"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        discourse_type = self.discourse_type[index]
        discourse = self.discourse[index]
        essay = self.essay[index]
        text = (
            discourse_type
            + " "
            + discourse
            + " "
            + self.tokenizer.sep_token
            + " "
            + essay
        )
        inputs = self.tokenizer.encode_plus(
            text, truncation=True, add_special_tokens=True, max_length=self.max_len
        )

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "target": self.targets[index],
        }


def get_loaders(fold):
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

    df = pd.read_csv(config["train_folds"])
    df_train = df[df.fold != fold].reset_index(drop=True)
    df_valid = df[df.fold == fold].reset_index(drop=True)

    train_dataset = FeedBackDataset(
        df_train, tokenizer=tokenizer, max_length=config["max_length"]
    )
    valid_dataset = FeedBackDataset(
        df_valid, tokenizer=tokenizer, max_length=config["max_length"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train_bs"],
        collate_fn=collate_fn,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["valid_bs"],
        collate_fn=collate_fn,
        num_workers=2,
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, valid_loader
