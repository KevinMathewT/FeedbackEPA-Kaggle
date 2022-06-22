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
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    history = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        train_epoch_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            accelerator=accelerator,
            epoch=epoch,
        )

        val_epoch_loss = valid_one_epoch(
            model=model,
            dataloader=valid_loader,
            criterion=criterion,
            accelerator=accelerator,
            epoch=epoch,
        )

        history["Train Loss"].append(train_epoch_loss)
        history["Valid Loss"].append(val_epoch_loss)

        # Log the metrics
        wandb.log({"Train Loss": train_epoch_loss})
        wandb.log({"Valid Loss": val_epoch_loss})

        # deep copy the model
        if val_epoch_loss <= best_epoch_loss:
            print(
                f"{b_}Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})"
            )
            best_epoch_loss = val_epoch_loss
            # run.summary["Best Loss"] = best_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = Path(config['weights_save']) / f"Loss-Fold-{fold}-{epoch}"
            # kaggle.api.dataset_initialize(Path(config['weights_save']))
            # kaggle.api.dataset_create_new(Path(config['weights_save']), public=False, )
            # torch.save(model.state_dict(), PATH)
            accelerator.save_state(PATH)
            # Save a model file from the current directory
            print(f"Model Saved to: {PATH}{sr_}")

        print()

    end = time.time()
    time_elapsed = end - start
    print(
        "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
            time_elapsed // 3600,
            (time_elapsed % 3600) // 60,
            (time_elapsed % 3600) % 60,
        )
    )
    print("Best Loss: {:.4f}".format(best_epoch_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history
