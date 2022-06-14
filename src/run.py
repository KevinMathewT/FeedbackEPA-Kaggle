import gc
from colorama import Fore, Back, Style

b_ = Fore.BLUE
y_ = Fore.YELLOW
sr_ = Style.RESET_ALL

import torch
from accelerate import Accelerator, DistributedType

from config import config

from .dataloader import get_loaders
from .model import get_model
from .loss_opt import get_criterion, get_optimizer, get_scheduler
from .trainer import get_trainer
from .utils import seed_everything

print(f"*** Training on folds: {config['tr_folds']} ***")

for fold in config['tr_folds']:
    seed_everything(config['seed'])

    print(f"{y_}====== Fold: {fold} ======{sr_}")
    # run = wandb.init(project='FeedBack',
    #                  config=CONFIG,
    #                  job_type='Train',
    #                  group=CONFIG['group'],
    #                  tags=[CONFIG['model_name'], f'{HASH_NAME}'],
    #                  name=f'{HASH_NAME}-fold-{fold}',
    #                  anonymous='must')

    accelerator = Accelerator()
    print(f"running on device: {accelerator.device}")
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

    # Create dataloaders, model, optimizer, scheduler, criterion etc
    model = get_model()
    train_loader, valid_loader = get_loaders(fold=fold)
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
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

    del model, history, train_loader, valid_loader
    _ = gc.collect()
    print()
