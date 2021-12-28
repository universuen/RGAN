import random

import torch
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.nn.functional import binary_cross_entropy
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from src import config, Logger
from src import models


class RGAN:

    def __init__(
            self,
            encoder: models.EncoderModel,
    ):
        self.logger = Logger(self.__class__.__name__)

        self.encoder = encoder
        self.generator = models.GeneratorModel().to(config.device)
        self.discriminator = models.DiscriminatorModel().to(config.device)
        self.classifier = models.ClassifierModel().to(config.device)

        self.generator_optimizer = Adam(
            params=self.generator.parameters(),
            lr=config.training.rgan.generator_lr,
            betas=(0.5, 0.9),
        )
        self.discriminator_optimizer = Adam(
            params=self.discriminator.parameters(),
            lr=config.training.rgan.discriminator_lr,
            betas=(0.5, 0.9),
        )
        self.classifier_optimizer = Adam(
            params=self.classifier.parameters(),
            lr=config.training.rgan.classifier_lr,
            betas=(0.5, 0.9),
        )

        self.statistics = {
            'classifier_loss': [],
            'classifier_hidden_loss': [],
            'discriminator_loss': [],
            'discriminator_hidden_loss': [],
            'generator_loss': [],
        }

    def train(self, minority_roulette_dataset: Dataset, majority_dataset: Dataset):
        self.logger.info('Started training')
        self.logger.debug(f'Using device: {config.device}')

        seed = random.choice(minority_roulette_dataset[:][0]).to(config.device)

        for _ in tqdm(range(config.training.rgan.epochs)):
            loss = 0
            for _ in range(config.training.rgan.classifier_loop_num):
                loss = self._train_classifier(majority_dataset, seed)
            self.statistics['classifier_loss'].append(loss)
            for _ in range(config.training.rgan.discriminator_loop_num):
                loss = self._train_discriminator(minority_roulette_dataset)
            self.statistics['discriminator_loss'].append(loss)
            for _ in range(config.training.rgan.generator_loop_num):
                loss = self._train_generator(len(minority_roulette_dataset), seed)
            self.statistics['generator_loss'].append(loss)

        self.classifier.eval()
        self.discriminator.eval()
        self.generator.eval()
        self._save_model()
        self._plot()
        self.logger.info(f'Finished training')

    def _train_discriminator(self, minority_dataset: Dataset) -> float:
        self.discriminator.zero_grad()
        x = minority_dataset[:][0].to(config.device)
        prediction_real = self.discriminator(x)
        loss_real = - prediction_real.mean()
        z = torch.randn(len(x), config.data.z_size, device=config.device)
        fake_x = self.generator(z).detach()
        prediction_fake = self.discriminator(fake_x)
        loss_fake = prediction_fake.mean()
        loss = loss_real + loss_fake
        loss.backward()
        self.discriminator_optimizer.step()
        return loss.item()

    def _train_classifier(self, majority_dataset: Dataset, seed: torch.Tensor) -> float:
        self.classifier.zero_grad()

        majorities = majority_dataset[:][0].to(config.device)
        seed = torch.stack([seed for _ in range(len(majorities))]).to(config.device)
        z, _, _ = self.encoder(seed)
        minorities = self.generator(z)

        minority_prediction = self.classifier(minorities).squeeze(dim=0)
        minority_loss = binary_cross_entropy(
            minority_prediction,
            torch.ones_like(minority_prediction),
        )
        majority_prediction = self.classifier(majorities).squeeze(dim=0)
        majority_loss = binary_cross_entropy(
            majority_prediction,
            torch.zeros_like(majority_prediction),
        )

        loss = (minority_loss + majority_loss) / 2
        loss.backward()
        self.classifier_optimizer.step()

    def _train_generator(self, x_len: int, seed: torch.Tensor) -> float:
        self.generator.zero_grad()
        cal_kl_div = torch.nn.KLDivLoss(reduction='batchmean')

        c_majority_hidden_output = self.classifier.hidden_output.detach()
        c_seed = torch.stack([seed for _ in range(len(c_majority_hidden_output))]).to(config.device)
        z, _, _ = self.encoder(c_seed)
        minorities = self.generator(z)
        self.classifier(minorities)
        c_minority_hidden_output = self.classifier.hidden_output.detach()
        c_majority_hidden_distribution = c_majority_hidden_output / c_majority_hidden_output.max()
        c_minority_hidden_distribution = c_minority_hidden_output / c_minority_hidden_output.max()
        c_hidden_loss = - cal_kl_div(
            input=c_minority_hidden_distribution,
            target=c_majority_hidden_distribution,
        ) * config.training.rgan.c_hl_lambda

        d_seed = torch.stack([seed for _ in range(x_len)]).to(config.device)
        z, _, _ = self.encoder(d_seed)
        fake_x = self.generator(z)
        d_real_x_hidden_output = self.discriminator.hidden_output.detach()
        d_final_output = self.discriminator(fake_x)
        d_fake_x_hidden_output = self.discriminator.hidden_output
        d_real_x_hidden_distribution = d_real_x_hidden_output / d_real_x_hidden_output.max()
        d_fake_x_hidden_distribution = d_fake_x_hidden_output / d_fake_x_hidden_output.max()
        d_hidden_loss = cal_kl_div(
            input=d_fake_x_hidden_distribution,
            target=d_real_x_hidden_distribution,
        ) * config.training.rgan.d_hl_lambda

        self.statistics['classifier_hidden_loss'].append(c_hidden_loss.item())
        self.statistics['discriminator_hidden_loss'].append(d_hidden_loss.item())
        loss = -d_final_output.mean() + c_hidden_loss + d_hidden_loss
        loss.backward()
        self.generator_optimizer.step()
        return loss.item()

    def _save_model(self):
        classifier_path = config.path.models / f'{self.__class__.__name__}_classifier.pt'
        torch.save(self.classifier.state_dict(), classifier_path)
        self.logger.debug(f'Saved classifier model at {classifier_path}')

        discriminator_path = config.path.models / f'{self.__class__.__name__}_discriminator.pt'
        torch.save(self.discriminator.state_dict(), discriminator_path)
        self.logger.debug(f'Saved discriminator model at {discriminator_path}')

        generator_path = config.path.models / f'{self.__class__.__name__}_generator.pt'
        torch.save(self.generator.state_dict(), generator_path)
        self.logger.debug(f'Saved generator model at {generator_path}')

    def _plot(self):
        sns.set()
        plt.title(f"{self.__class__.__name__} Generator and Discriminator Loss During Training")
        plt.plot(self.statistics['generator_loss'], label="Generator")
        plt.plot(self.statistics['discriminator_loss'], label="Discriminator")
        plt.plot(self.statistics['classifier_loss'], label="Classifier")
        plt.plot(self.statistics['discriminator_hidden_loss'], label="Discriminator Hidden")
        plt.plot(self.statistics['classifier_hidden_loss'], label="Classifier Hidden")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plot_path = config.path.plots / f'{self.__class__.__name__}_Loss.png'
        plt.savefig(fname=str(plot_path))
        plt.clf()
        self.logger.debug(f'Saved plot at {plot_path}')

    def load_model(self):
        discriminator_path = config.path.models / f'{self.__class__.__name__}_discriminator.pt'
        self.discriminator.load_state_dict(
            torch.load(discriminator_path)
        )
        self.discriminator.to(config.device)
        self.discriminator.eval()
        self.logger.debug(f'Loaded discriminator model from {discriminator_path}')

        classifier_path = config.path.models / f'{self.__class__.__name__}_classifier.pt'
        self.classifier.load_state_dict(
            torch.load(classifier_path)
        )
        self.classifier.to(config.device)
        self.classifier.eval()
        self.logger.debug(f'Loaded classifier model from {classifier_path}')

        generator_path = config.path.models / f'{self.__class__.__name__}_generator.pt'
        self.generator.load_state_dict(
            torch.load(generator_path)
        )
        self.generator.to(config.device)
        self.generator.eval()
        self.logger.debug(f'Loaded generator model from {generator_path}')
