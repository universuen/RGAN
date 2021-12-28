import torch

from src.config import (
    training,
    data,
    dataset,
    logger,
    path,
)

# random seed
seed: int = 0

# pytorch device
device = 'auto'

if device == 'auto':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
