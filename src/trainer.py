import gc
import time
import copy
import wandb
from pathlib import Path
from collections import defaultdict
from colorama import Fore, Back, Style

import numpy as np

import torch

from .train import train_one_epoch, valid_one_epoch

from config import config

b_ = Fore.BLUE
sr_ = Style.RESET_ALL


class Validator:
    def __init__(self, model):
        self.valid_freq_per_epoch = config["valid_freq_per_epoch"]
        self.best_epoch_loss = np.inf
        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.history = defaultdict(list)

    def validate(self, model, valid_loader, criterion, accelerator, epoch, index, fold):
        print(
            "\t" + ("*" * 5) + f" validation: epoch {epoch}, index {index} " + ("*" * 5)
        )
        val_epoch_loss = valid_one_epoch(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
            accelerator=accelerator,
            epoch=epoch,
        )
        self.history["Valid Loss"].append(val_epoch_loss)
        wandb.log({"valid epoch loss": val_epoch_loss})

        if val_epoch_loss <= self.best_epoch_loss:
            print(
                "\t"
                + f"{b_}Validation Loss Improved ({self.best_epoch_loss} ---> {val_epoch_loss})"
            )
            self.best_epoch_loss = val_epoch_loss
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            self.best_model_wts = copy.deepcopy(unwrapped_model.state_dict())
            PATH = Path(config["weights_save"]) / f"Loss-Fold-{fold}.bin"

            accelerator.save(unwrapped_model.state_dict(), PATH)
            # Save a model file from the current directory
            print("\t" + f"Model Saved to: {PATH}{sr_}")

            del unwrapped_model
            _ = gc.collect()
        
        


def get_trainer(
    model,
    train_loader,
    valid_loader,
    optimizer,
    scheduler,
    criterion,
    accelerator,
    num_epochs,
    fold,
):
    # To automatically log gradients
    wandb.watch(model, log_freq=100)

    start = time.time()
    validator = Validator(model)

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        train_epoch_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            accelerator=accelerator,
            epoch=epoch,
            validator=validator,
            fold=fold,
        )

        validator.history["Train Loss"].append(train_epoch_loss)
        wandb.log({"train epoch loss": train_epoch_loss})

    end = time.time()
    time_elapsed = end - start
    print(
        "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
            time_elapsed // 3600,
            (time_elapsed % 3600) // 60,
            (time_elapsed % 3600) % 60,
        )
    )
    print("Best Loss: {:.4f}".format(validator.best_epoch_loss))

    # load best model weights
    model.load_state_dict(validator.best_model_wts)

    return model, validator.history
