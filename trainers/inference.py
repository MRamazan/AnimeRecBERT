from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks

import torch.nn as nn
import time
import torch


def load_checkpoint(model, optimizer=None, path='checkpoint.pth', map_location='cuda'):
    checkpoint = torch.load(path, map_location=map_location)

    # Model ağırlıklarını yükle
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Model ağırlıkları yüklendi.")

    # Optimizer varsa onu da yükle
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✅ Optimizer durumu yüklendi.")

    # Epoch veya loss gibi ek bilgiler varsa:
    epoch = checkpoint.get('epoch', None)
    loss = checkpoint.get('loss', None)

    return model, optimizer, epoch, loss

