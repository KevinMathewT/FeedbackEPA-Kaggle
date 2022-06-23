import gc
import wandb
from time import time
from tqdm import tqdm

import torch
import torch.nn as nn

from config import config


def train_one_epoch(
    model, dataloader, optimizer, scheduler, criterion, accelerator, epoch
):
    model.train()

    dataset_size = 0
    running_loss = 0.0
    running_bce_loss = 0.0
    st = time()

    # bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in enumerate(dataloader): # bar:
        ids = data["input_ids"]
        mask = data["attention_mask"]
        targets = data["target"]

        batch_size = ids.size(0)

        outputs = model(ids, mask)

        loss = criterion(outputs, targets)
        loss = loss / config["n_accumulate"]
        bce_loss = nn.CrossEntropyLoss()(outputs, targets)
        accelerator.backward(loss)
        wandb.log({"train step loss": loss})

        if (step + 1) % config["n_accumulate"] == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += loss.item() * batch_size
        running_bce_loss += bce_loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        epoch_bce_loss = running_bce_loss / dataset_size

        if step == 0 or (step + 1) % config['freq'] == 0 or step == len(dataloader) - 1:
            accelerator.print(f"[{epoch}/{config['epochs']}][{str(step + 1):5s}/{len(dataloader)}] train loss: {epoch_loss:1.10f} | bce_loss: {epoch_bce_loss:1.10f} | lr: {optimizer.param_groups[0]['lr']:1.10f} | time: {time() - st:1.1f}s")
        
        # optimizer.param_groups[0]["lr"]
        # bar.set_postfix(
        #     Epoch=epoch, Train_Loss=epoch_loss, LR=optimizer.param_groups[0]["lr"]
        # )
    gc.collect()

    return epoch_loss


@torch.no_grad()
def valid_one_epoch(model, dataloader, criterion, accelerator, epoch):
    model.eval()

    dataset_size = 0
    running_loss = 0.0
    running_bce_loss = 0.0
    st = time()

    # bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in enumerate(dataloader): # bar:
        ids = data["input_ids"]
        mask = data["attention_mask"]
        targets = data["target"]

        batch_size = ids.size(0)

        outputs = model(ids, mask)

        loss = criterion(outputs, targets)
        bce_loss = nn.CrossEntropyLoss()(outputs, targets)
        wandb.log({"valid step loss": loss})

        running_loss += loss.item() * batch_size
        running_bce_loss += bce_loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        epoch_bce_loss = running_bce_loss / dataset_size

        if step == 0 or (step + 1) % config['freq'] == 0 or step == len(dataloader) - 1:
            accelerator.print(f"[{epoch}/{config['epochs']}][{str(step + 1):5s}/{len(dataloader)}] valid loss: {epoch_loss:1.10f} | bce_loss: {epoch_bce_loss:1.10f} | time: {time() - st:1.1f}s")
        # bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss)

    gc.collect()

    return epoch_loss
