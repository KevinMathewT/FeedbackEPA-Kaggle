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
    running_ce_loss = 0.0
    st = time()

    # bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in enumerate(dataloader):  # bar:
        ids = data["input_ids"]
        mask = data["attention_mask"]
        targets = data["target"]

        batch_size = ids.size(0)

        outputs = model(ids, mask)

        loss = criterion(outputs, targets)
        loss = loss / config["n_accumulate"]
        ce_loss = nn.CrossEntropyLoss()(outputs, targets)
        accelerator.backward(loss)
        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), config["max_grad_norm"]
        )
        wandb.log({"train step loss": loss})

        if (step + 1) % config["n_accumulate"] == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += loss.item() * batch_size
        running_ce_loss += ce_loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        epoch_ce_loss = running_ce_loss / dataset_size

        if step == 0 or (step + 1) % config["freq"] == 0 or step == len(dataloader) - 1:
            accelerator.print(
                f"[{epoch}/{config['epochs']}][{str(step + 1):5s}/{len(dataloader)}] train loss: {epoch_loss:1.10f} | ce_loss: {epoch_ce_loss:1.10f} | lr: {optimizer.param_groups[0]['lr']:1.10f} | grad norm: {grad_norm:1.4f} | time: {time() - st:1.1f}s"
            )

    gc.collect()

    return epoch_loss


@torch.no_grad()
def valid_one_epoch(model, dataloader, criterion, accelerator, epoch):
    model.eval()

    dataset_size = 0
    running_loss = 0.0
    running_ce_loss = 0.0
    st = time()

    # bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in enumerate(dataloader):  # bar:
        ids = data["input_ids"]
        mask = data["attention_mask"]
        targets = data["target"]

        batch_size = ids.size(0)

        outputs = model(ids, mask)

        loss = criterion(outputs, targets)
        ce_loss = nn.CrossEntropyLoss()(outputs, targets)
        wandb.log({"valid step loss": loss})

        running_loss += loss.item() * batch_size
        running_ce_loss += ce_loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        epoch_ce_loss = running_ce_loss / dataset_size

        if step == 0 or (step + 1) % config["freq"] == 0 or step == len(dataloader) - 1:
            accelerator.print(
                f"[{epoch}/{config['epochs']}][{str(step + 1):5s}/{len(dataloader)}] valid loss: {epoch_loss:1.10f} | ce_loss: {epoch_ce_loss:1.10f} | time: {time() - st:1.1f}s"
            )

    gc.collect()

    return epoch_loss
