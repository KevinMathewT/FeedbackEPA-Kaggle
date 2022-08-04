import os
import json
import codecs
from glob import glob
import argparse
from pathlib import Path
from typing import Tuple
from text_unidecode import unidecode

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch

from datasets import load_dataset

import tokenizers
import transformers
from transformers import AutoTokenizer, AutoConfig
from transformers import DataCollatorForLanguageModeling, AutoModelForMaskedLM, Trainer
from transformers import TrainingArguments
from transformers.utils import logging

from config import config
from src.utils import seed_everything

seed_everything(42)
logging.set_verbosity_info()
logger = logging.get_logger(__name__)
logger.info("INFO")
logger.warning("WARN")
KAGGLE_ENV = True if "KAGGLE_URL_BASE" in set(os.environ.keys()) else False


def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end


codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)


def resolve_encodings_and_normalize(text: str) -> str:
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text


print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_patient_notes_not_used_train():
    essay_fps = glob(config["mlm_f1_train_base"] + "*.txt") + glob(
        config["mlm_f2_train_base"] + "*.txt"
    )
    print(f"essay count: {len(essay_fps)}")

    essays = []
    for fp in tqdm(essay_fps):
        essays.append(resolve_encodings_and_normalize(open(Path(fp), "r").read()))

    essays = np.array(essays)
    np.random.shuffle(essays)
    essays = essays.tolist()
    return (
        essays[: -int(len(essays) * config["mlm_test_split"])],
        essays[-int(len(essays) * config["mlm_test_split"]) :],
    )


def tokenize_function(examples):
    return tokenizer(examples["text"])


def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], use_fast=True)
    return tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=config['model_name'], required=False)
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
    )
    parser.add_argument("--seed", type=int, default=0, required=False)
    parser.add_argument("--debug", action="store_true", required=False)
    parser.add_argument("--exp_num", type=str, required=True)
    parser.add_argument("--param_freeze", action="store_true", required=False)
    parser.add_argument("--num_train_epochs", type=int, default=5, required=False)
    parser.add_argument("--batch_size", type=int, default=8, required=False)
    parser.add_argument("--lr", type=float, default=2e-5, required=False)
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1, required=False
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    train_text_list, valid_text_list = get_patient_notes_not_used_train()
    print(f"train text list lenght: {len(train_text_list)}")
    print(f"valid text list lenght: {len(valid_text_list)}")

    if args.debug:
        train_text_list = train_text_list[:10]
        valid_text_list = valid_text_list[:10]
        args.batch_seize = 1

    def get_text(df):
        text_list = []
        for text in tqdm(df["pn_history"]):
            if len(text) < 30:
                pass
            else:
                text_list.append(text)
        return text_list

    mlm_train_json_path = Path(config["mlm_gen"]) / "train_mlm.json"
    mlm_valid_json_path = Path(config["mlm_gen"]) / "valid_mlm.json"

    for json_path, list_ in zip(
        [mlm_train_json_path, mlm_valid_json_path], [train_text_list, valid_text_list]
    ):
        with open(str(json_path), "w") as f:
            for sentence in list_:
                row_json = {"text": sentence}
                json.dump(row_json, f)
                f.write("\n")

    datasets = load_dataset(
        "json",
        data_files={
            "train": str(mlm_train_json_path),
            "valid": str(mlm_valid_json_path),
        },
    )

    if mlm_train_json_path.is_file():
        mlm_train_json_path.unlink()
    if mlm_valid_json_path.is_file():
        mlm_valid_json_path.unlink()
    print(datasets["train"][:2])

    tokenizer = get_tokenizer(args)

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=["text"],
        batch_size=args.batch_size,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15
    )

    if args.model_name:
        print("model_name:", args.model_name)
        model_name = args.model_name
    else:
        print("model_path:", args.model_path)
        model_name = args.model_path

    model_config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name, config=model_config)
    model.gradient_checkpointing_enable()


    if args.param_freeze:
        # if freeze, Write freeze settings here

        # deberta-v3-large
        # model.deberta.embeddings.requires_grad_(False)
        # model.deberta.encoder.layer[:12].requires_grad_(False)

        # deberta-large
        model.deberta.embeddings.requires_grad_(False)
        model.deberta.encoder.layer[:24].requires_grad_(False)

        for name, p in model.named_parameters():
            print(name, p.requires_grad)

    if args.debug:
        save_steps = 100
        args.num_train_epochs = 1
    else:
        save_steps = 100000000

    training_args = TrainingArguments(
        output_dir=config['weights_save'],
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        weight_decay=0.01,
        save_strategy="no",
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        # report_to="wandb",
        run_name=f"output-mlm-{args.exp_num}",
        # logging_dir='./logs',
        lr_scheduler_type="cosine",
        warmup_ratio=0.2,
        fp16=True,
        logging_steps=500,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        data_collator=data_collator,
        # optimizers=(optimizer, scheduler)
    )

    trainer.train()

    if args.model_name == "microsoft/deberta-xlarge":
        model_name = "deberta-xlarge"
    elif args.model_name == "microsoft/deberta-large":
        model_name = "deberta-large"
    elif args.model_name == "microsoft/deberta-base":
        model_name = "deberta-base"
    elif args.model_path == "../input/deberta-v3-large/deberta-v3-large/":
        model_name = "deberta-v3-large"
    elif args.model_name == "microsoft/deberta-v2-xlarge":
        model_name = "deberta-v2-xlarge"
    trainer.model.save_pretrained(
        Path(config["mlm_gen"]) / f"{args.exp_num}_mlm_{model_name}"
    )
