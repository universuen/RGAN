import random
from typing import Callable

import numpy as np
import torch

from src.logger import Logger
from src.datasets import MinorityDataset, MajorityDataset


class MinorityRouletteDataset(MinorityDataset):
    def __init__(
            self,
            training: bool = True,
            transform: Callable = None,
            target_transform: Callable = None
    ):
        super().__init__(training, transform, target_transform)
        self.logger = Logger(self.__class__.__name__)

        minorities = MinorityDataset(training, transform, target_transform)[:][0]
        majorities = MajorityDataset(training, transform, target_transform)[:][0]
        dist = np.zeros([len(minorities), len(majorities)])

        # calculate distances
        for i, minority_item in enumerate(minorities):
            for j, majority_item in enumerate(majorities):
                dist[i][j] = torch.norm(minority_item - majority_item, p=2)

        self.fits = 1 / dist.min(axis=1, initial=None)
        self.fits = self.fits / self.fits.sum()

    def get_roulette_choices(self, size: int) -> (torch.Tensor, torch.Tensor):
        if size == len(self):
            self.logger.warning('Roulette failed! Choices size reaches the max size.')
        chosen_indices = np.random.choice(
            list(range(len(self))),
            size=size,
            replace=False,
            p=self.fits,
        )
        return self.features[chosen_indices], self.labels[chosen_indices]

