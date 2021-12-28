import context

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.datasets import make_classification

from src import config, utils
from src.datasets import FullDataset, MinorityRouletteDataset

RATIO = 0.1

if __name__ == '__main__':
    x, y = make_classification(1000, n_features=2, n_informative=2, n_redundant=0, random_state=config.seed)
    np.save(config.path.processed_datasets / 'training_features.npy', x)
    np.save(config.path.processed_datasets / 'training_labels.npy', y)

    full_dataset = FullDataset()

    features = full_dataset[:][0].numpy()
    types = []

    for _, i in full_dataset:
        if i.item() == 0:
            types.append('Majority')
        else:
            types.append('Minority')

    r_dataset = MinorityRouletteDataset()
    utils.set_random_state()
    r_features, r_labels = r_dataset.get_roulette_choices(int(RATIO * len(r_dataset)))
    features = np.append(features, r_features, axis=0)
    types.extend(['Roulette' for _ in r_labels])

    features = TSNE(
        learning_rate='auto',
        init='random',
        random_state=config.seed,
    ).fit_transform(features)

    df = pd.DataFrame(
        {
            'f_x': features[:, 0],
            'f_y': features[:, 1],
            'type': types,
        }
    )

    sns.set()
    ax = sns.scatterplot(
        data=df,
        x='f_x',
        y='f_y',
        hue='type',
        alpha=0.7,
    )
    ax.set(xticklabels=[])
    ax.set(xlabel=None)
    ax.set(yticklabels=[])
    ax.set(ylabel=None)
    ax.set(title='Roulette Result')
    plt.savefig(config.path.plots / 'Roulette_Result.jpg')
    plt.show()
