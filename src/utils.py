import random

import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

from src import config


def set_random_state(seed: int = None) -> None:
    if seed is None:
        seed = config.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_weights(layer: nn.Module):
    layer_name = layer.__class__.__name__
    if 'Linear' in layer_name:
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0)
    elif layer_name == 'BatchNorm1d':
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)


def preprocess_data(file_name: str) -> (np.ndarray, np.ndarray):
    set_random_state()
    # concatenate the file path
    file_path = config.path.raw_datasets / file_name
    # calculate skip rows
    skip_rows = 0
    with open(file_path, 'r') as f:
        while True:
            line = f.readline()
            if line[0] != '@':
                break
            else:
                skip_rows += 1
    # read raw data
    df = pd.read_csv(file_path, sep=',', skiprows=skip_rows, header=None)
    np_array = df.to_numpy()
    np.random.shuffle(np_array)
    # partition labels and features
    labels = np_array[:, -1].copy()
    features = np_array[:, :-1].copy()
    # digitize labels
    for i, _ in enumerate(labels):
        labels[i] = labels[i].strip()
    labels[labels[:] == 'positive'] = 1
    labels[labels[:] == 'negative'] = 0
    labels = labels.astype('int')
    # normalize features
    features = normalize(features)
    config.data.x_size = features.shape[1]
    return features, labels


def save_partitioned_dataset(
        features: np.ndarray,
        labels: np.ndarray,
) -> None:
    training_features, test_features, training_labels, test_labels = train_test_split(
        features,
        labels,
        train_size=config.dataset.training_ratio,
        random_state=config.seed,
    )
    np.save(str(config.path.processed_datasets / 'training_features.npy'), training_features)
    np.save(str(config.path.processed_datasets / 'test_features.npy'), test_features)
    np.save(str(config.path.processed_datasets / 'training_labels.npy'), training_labels)
    np.save(str(config.path.processed_datasets / 'test_labels.npy'), test_labels)


def prepare_dataset(file_name: str) -> None:
    save_partitioned_dataset(*preprocess_data(file_name))


def get_final_test_metrics(statistics: dict):
    metrics = dict()
    for name, values in statistics.items():
        if name == 'Loss':
            continue
        else:
            metrics[name] = values[-1]
    return metrics


def normalize(x: torch.Tensor) -> torch.Tensor:
    return minmax_scale(x)
