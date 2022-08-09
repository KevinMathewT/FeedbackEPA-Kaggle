import gc
import wandb
from colorama import Fore, Back, Style

b_ = Fore.BLUE
y_ = Fore.YELLOW
sr_ = Style.RESET_ALL

import torch
from accelerate import Accelerator, DistributedType

from config import config

from .dataloader import get_loaders
from .model import get_model
from .optimizer import get_optimizer, get_scheduler
from .criterion import get_criterion
from .trainer import get_trainer
from .utils import dump_tensors, seed_everything


def run(index):
    print(f"*** Training on folds: {config['tr_folds']} ***")

    for fold in config["tr_folds"]:
        accelerator = Accelerator(cpu=config["cpu"], mixed_precision=config["amp"])
        accelerator.print(index)
        accelerator.print(f"{y_}====== Fold: {fold} ======{sr_}")

        run = wandb.init(project='FeedBackPEA',
                         config=config,
                         job_type='Train')

        accelerator.print(f"running on device: {accelerator.device}")
        if torch.cuda.is_available():
            print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

        # Create dataloaders, model, optimizer, scheduler, criterion etc
        model = get_model()
        train_loader, valid_loader = get_loaders(fold=fold, accelerator=accelerator)
        optimizer = get_optimizer(model)
        scheduler = get_scheduler(optimizer, num_training_steps=(len(train_loader) * config['epochs']))
        criterion = get_criterion()

        model, train_loader, valid_loader, optimizer, scheduler = accelerator.prepare(
            model, train_loader, valid_loader, optimizer, scheduler
        )

        model, history = get_trainer(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            accelerator=accelerator,
            num_epochs=config["epochs"],
            fold=fold,
        )

        # run.finish()

        del model, history, train_loader, valid_loader, optimizer, scheduler, criterion, accelerator
        _ = gc.collect()
        torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    seed_everything(config["seed"])

    if config["tpu"]:
        import torch_xla.distributed.xla_multiprocessing as xmp

        xmp.spawn(run, args=())
    else:
        run(0)
