import context

from src import utils
from src.datasets import FullDataset
from src import Classifier


FILE_NAME = 'segment0.dat'

if __name__ == '__main__':

    # prepare dataset
    utils.prepare_dataset(FILE_NAME)

    # normally train
    utils.set_random_state()
    classifier = Classifier('Test_Normal_Train')
    classifier.train(
        training_dataset=FullDataset(training=True),
        test_dateset=FullDataset(training=False),
    )
    for name, value in utils.get_final_test_metrics(classifier.statistics).items():
        print(f'{name:<10}:{value:>10}')
