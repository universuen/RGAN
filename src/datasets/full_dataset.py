from typing import Callable

import torch
import numpy as np
from torch.utils.data import Dataset as Base

from src import config


class FullDataset(Base):
    def __init__(
            self,
            training: bool = True,
            transform: Callable = None,
            target_transform: Callable = None,
    ):
        if training is True:
            features_path = config.path.processed_datasets / 'training_features.npy'
            labels_path = config.path.processed_datasets / 'training_labels.npy'
        else:
            features_path = config.path.processed_datasets / 'test_features.npy'
            labels_path = config.path.processed_datasets / 'test_labels.npy'

        a = np.load(labels_path)
        self.features = torch.from_numpy(
            np.load(str(features_path))
        ).float()
        self.labels = torch.from_numpy(
            np.load(str(labels_path))
        ).float()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        feature = self.features[item]
        label = self.labels[item]
        if self.transform:
            feature = self.transform(feature)
        if self.target_transform:
            label = self.target_transform(label)
        return feature, label
