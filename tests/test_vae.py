
import context

from src import utils, config
from src.datasets import MinorityDataset
from src import VAE


FILE_NAME = 'segment0.dat'

if __name__ == '__main__':
    # prepare dataset
    utils.prepare_dataset(FILE_NAME)
    # train
    dataset = MinorityDataset(training=True)
    utils.set_random_state()
    vae = VAE()
    vae.train(dataset=dataset)
    # test
    x = dataset[:3][0].to(config.device)
    z, mu, sigma = vae.encoder(x)
    print(mu.mean().mean().item())
    print(sigma.mean().mean().item())
