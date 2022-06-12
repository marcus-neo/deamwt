"""Module Containing the Callback Classes."""

import random
from typing import Optional, Union
import warnings
import math


import numpy as np
import torch
import pytorch_lightning as pl
import torchvision

import matplotlib.pyplot as plt

from utils.sampler import Sampler
from utils.train_type import TrainType
from utils.preprocessing import db1_multires_analysis
from utils.errors import checker_train_type_decom

# Ignore torchvision UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)


class GenerateCallback(pl.Callback):

    """Generate Callback Class."""

    def __init__(
        self,
        batch_size=8,
        vis_steps=8,
        num_steps=4096,
        every_n_epochs=5,
        train_type=TrainType.DECOMPOSED,
        decom_level: Optional[Union[int, None]] = None,
    ):
        """Initialise the Callback Class."""
        super().__init__()
        self.batch_size = batch_size  # Number of images to generate
        self.vis_steps = (
            vis_steps  # Number of steps within generation to visualize
        )
        self.num_steps = num_steps  # Number of steps to take during generation
        # Only save those images every N epochs
        # (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs
        self.train_type = train_type
        self.decom_level = decom_level
        checker_train_type_decom(self.decom_level, self.train_type)

    def on_epoch_end(self, trainer, pl_module):
        """Run the callback at the end of each epoch."""
        # Skip for all other epochs
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Generate fake data
            fakes_per_step = self.generate_fakes(pl_module)
            # Plot and add to tensorboard
            # for each step
            for i in range(fakes_per_step.shape[1]):
                # obtain the fake data for that step
                step_size = self.num_steps // self.vis_steps
                fakes_to_plot = fakes_per_step[step_size - 1 :: step_size, i]
                sigs_to_plot = (
                    torch.sum(fakes_to_plot, dim=3)
                    .detach()
                    .cpu()
                    .numpy()
                    .squeeze()
                )
                fig, axs = plt.subplots(sigs_to_plot.shape[0], 1)
                for ind, axis in enumerate(axs):
                    axis.plot(sigs_to_plot[ind])
                trainer.logger.experiment.add_figure(
                    f"generation_{i}", fig, global_step=trainer.current_epoch
                )

    def generate_fakes(self, pl_module):
        """Generate Fake Data from the Energy Based Model."""
        pl_module.eval()
        if self.train_type == TrainType.PURE:
            start_imgs = torch.rand(
                (self.batch_size,) + pl_module.hparams["img_shape"]
            ).to(pl_module.device)
            start_imgs = start_imgs * 2 - 1
        elif self.train_type == TrainType.DECOMPOSED:
            randint = random.randint(0, 100)
            start_sigs = np.expand_dims(
                np.array(
                    [
                        db1_multires_analysis(
                            np.array(
                                [
                                    math.sin(
                                        2
                                        * math.pi
                                        * random.random()
                                        * (i + randint)
                                    )
                                    for i in range(
                                        pl_module.hparams["img_shape"][1]
                                    )
                                ]
                            ),
                            level=self.decom_level,
                        )
                        for _ in range(self.batch_size)
                    ]
                ),
                axis=1,
            )
            start_imgs = torch.from_numpy(start_sigs).to(pl_module.device)

        torch.set_grad_enabled(
            True
        )  # Tracking gradients for sampling necessary
        imgs_per_step = Sampler.generate_samples(
            pl_module.cnn,
            start_imgs,
            steps=self.num_steps,
            step_size=10,
            return_img_per_step=True,
        )
        torch.set_grad_enabled(False)
        pl_module.train()
        return imgs_per_step


class SamplerCallback(pl.Callback):

    """Sampler Callback Class."""

    def __init__(self, num_imgs=32, every_n_epochs=5):
        """Initialise the Callback."""
        super().__init__()
        self.num_imgs = num_imgs  # Number of images to plot
        # Only save those images every N epochs
        # (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, trainer, pl_module):
        """Run the callback at the end of each epoch."""
        if trainer.current_epoch % self.every_n_epochs == 0:
            exmp_imgs = torch.cat(
                random.choices(pl_module.sampler.examples, k=self.num_imgs),
                dim=0,
            )
            grid = torchvision.utils.make_grid(
                exmp_imgs, nrow=4, normalize=True, range=(-1, 1)
            )
            trainer.logger.experiment.add_image(
                "sampler", grid, global_step=trainer.current_epoch
            )


class OutlierCallback(pl.Callback):

    """Outlier Callback Class."""

    def __init__(self, batch_size=1024):
        """Initialise the Callback class."""
        super().__init__()
        self.batch_size = batch_size

    def on_epoch_end(self, trainer, pl_module):
        """Run the callback at the end of each epoch."""
        with torch.no_grad():
            pl_module.eval()
            rand_imgs = torch.rand(
                (self.batch_size,) + pl_module.hparams["img_shape"]
            ).to(pl_module.device)
            rand_imgs = rand_imgs * 2 - 1.0
            rand_out = pl_module.cnn(rand_imgs).mean()
            pl_module.train()

        trainer.logger.experiment.add_scalar(
            "rand_out", rand_out, global_step=trainer.current_epoch
        )
