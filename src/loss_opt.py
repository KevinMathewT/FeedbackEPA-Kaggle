import torch.nn as nn
from torch.optim import lr_scheduler

from transformers import AdamW

from config import config

def get_criterion():
    return nn.CrossEntropyLoss()

def get_optimizer(model):
    return AdamW(model.parameters(), lr=config['lr'], weight_decay=config['wd'])

def get_scheduler(optimizer):
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['T_max'], 
                                                   eta_min=config['min_lr'])
    elif config['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=config['T_0'], 
                                                             eta_min=config['min_lr'])
    elif config['scheduler'] == "None":
        return None
        
    return scheduler

