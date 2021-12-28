import context

import torch

from src import utils, config
from src.datasets import MinorityDataset
from src import RGAN


FILE_NAME = 'segment0.dat'

if __name__ == '__main__':
    # prepare dataset
    utils.prepare_dataset(FILE_NAME)
    # train
    dataset = MinorityDataset(training=True)
    utils.set_random_state()
    gan = RGAN()
    gan.train(dataset=dataset)
    # test
    gan.load_model()
    z = torch.randn(1, config.data.z_size, device=config.device)
    print(gan.generator(z))
