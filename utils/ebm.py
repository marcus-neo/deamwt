"""Module containing the Deep Energy Based Model."""
import torch
from torch import optim
from torchinfo import summary
import pytorch_lightning as pl
from utils.errors import InternalModelError

from utils.model import CNNModel
from utils.sampler import Sampler
from utils.set_device import set_device


device = set_device()


# pylint: disable=too-many-ancestors
class DeepEnergyModel(pl.LightningModule):

    """Deep Energy Based Model."""

    # pylint: disable=unused-argument,invalid-name
    def __init__(
        self, img_shape, batch_size, alpha=0.1, lr=1e-4, beta1=0.0, **CNN_args
    ):
        """Initialize the model."""
        super().__init__()
        self.save_hyperparameters()

        self.cnn = CNNModel(img_shape, **CNN_args)
        summary(self.cnn, ((batch_size,) + img_shape))
        self.sampler = Sampler(
            self.cnn,
            img_shape=img_shape,
            sample_size=batch_size,
            device=device,
        )
        # Perform sanity check on the model
        print("Performing Sanity Check on the Internal Neural Network...")
        try:
            example_zeros_array = torch.zeros(1, *img_shape)
            example_input_array = torch.rand_like(example_zeros_array)
            sample_cnn = CNNModel(img_shape, **CNN_args)
            sample_cnn(example_zeros_array)
            sample_cnn(example_input_array)
            del example_input_array
            del example_zeros_array
            del sample_cnn
        except Exception as e:
            raise InternalModelError from e
        print("Internal Neural Network Sanity Check Passed.")

    def forward(self, *args, **kwargs):
        """Overwrite abstract forward class."""
        return self.cnn(args[0])

    def configure_optimizers(self):
        """Overwrite abstract configure_optimizers class."""
        # Energy models can have issues with momentum as the loss surfaces
        # changes with its parameters.
        # Hence, we set it to 0 by default.
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, 0.999),
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 1, gamma=0.97
        )  # Exponential decay over epochs
        return [optimizer], [scheduler]

    def training_step(self, *args, **kwargs):
        """Overwrite abstract training_step class."""
        # We add minimal noise to the original images to prevent the model
        # from focusing on purely "clean" inputs
        real_imgs = args[0]
        small_noise = torch.randn_like(real_imgs) * 0.005
        real_imgs.add_(small_noise).clamp_(min=-1.0, max=1.0)

        # Obtain samples
        fake_imgs = self.sampler.sample_new_exmps(steps=60, step_size=10)

        # Predict energy score for all images
        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)

        # Calculate losses
        reg_loss = self.hparams.alpha * (real_out**2 + fake_out**2).mean()
        cdiv_loss = fake_out.mean() - real_out.mean()
        loss = reg_loss + cdiv_loss

        # Logging
        self.log("loss", loss)
        self.log("loss_regularization", reg_loss)
        self.log("loss_contrastive_divergence", cdiv_loss)
        self.log("metrics_avg_real", real_out.mean())
        self.log("metrics_avg_fake", fake_out.mean())
        return loss

    def validation_step(self, *args, **kwargs):
        """Overwrite abstract validation_step class."""
        # For validating, we calculate the contrastive divergence between
        # purely random images and unseen examples
        # Note that the validation/test step of energy-based models depends
        # on what we are interested in the model
        real_imgs = args[0]
        fake_imgs = torch.rand_like(real_imgs) * 2 - 1

        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)

        cdiv = fake_out.mean() - real_out.mean()
        self.log("val_contrastive_divergence", cdiv)
        self.log("val_fake_out", fake_out.mean())
        self.log("val_real_out", real_out.mean())

    def predict_dataloader(self, *args, **kwargs):
        """Overwrite abstract predict_dataloader class."""

    def test_dataloader(self, *args, **kwargs):
        """Overwrite abstract test_dataloader class."""

    def train_dataloader(self, *args, **kwargs):
        """Overwrite abstract train_dataloader class."""

    def val_dataloader(self, *args, **kwargs):
        """Overwrite abstract val_dataloader class."""
