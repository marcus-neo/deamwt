"""Standard Train Script."""
# Standard libraries
import argparse
import os
from typing import Union, Optional

# Matplotlib
import matplotlib.pyplot as plt

# PyTorch
import torch
from torch.utils import data

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from utils.ebm import DeepEnergyModel
from utils.callbacks import (
    GenerateCallback,
    SamplerCallback,
    OutlierCallback,
)
from utils.set_device import set_device
from utils.errors import WindowDecomMismatchError
from utils.train_type import TrainType
from utils.datasets import TrainTestData, DatasetTrainVal


# pylint: disable=too-many-statements,too-many-locals,redefined-outer-name
def train_loop(
    train_type: TrainType,
    window: int,
    decom_level: Union[int, None],
    dataset_path: str,
    checkpoint_path: str,
    project_name: str,
    batch_size: int,
    seed: Optional[int] = 42,
) -> None:
    """Train the model on the train set."""
    train_type_str = train_type.name.lower()
    if train_type == TrainType.DECOMPOSED:
        if decom_level is None:
            raise WindowDecomMismatchError()
        logger_path = os.path.join(
            "logs",
            f"window_{window}",
            f"decom_{decom_level}",
            train_type_str,
        )
    elif train_type == TrainType.PURE:
        if decom_level is not None:
            raise WindowDecomMismatchError()
        logger_path = os.path.join(
            "logs",
            f"window_{window}",
            train_type_str,
        )
    logger = TensorBoardLogger(
        logger_path,
        project_name.replace("_", " ").capitalize(),
    )

    # Setting the seed
    pl.seed_everything(seed)

    # Ensure that all operations are deterministic on GPU (if used) for
    # reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    device = set_device()
    # Transformations applied on each image
    # => make them a tensor and normalize between -1 and 1
    print("Initialising Datasets...")
    all_data = TrainTestData(dataset_path, window)
    train_set = DatasetTrainVal(
        train_type=train_type,
        df_list=all_data.train_df_list,
        train_val=all_data.train,
        decom_level=decom_level,
    )
    validation_set = DatasetTrainVal(
        train_type=train_type,
        df_list=all_data.train_df_list,
        train_val=all_data.val,
        decom_level=decom_level,
    )
    print("Preparing Dataloaders...")
    # We define a set of data loaders that we can use for various purposes
    # later. Note that for actually training a model, we will use different
    # data loaders with a lower batch size.
    train_loader = data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )
    validation_loader = data.DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )

    def train_model(**kwargs):
        # Create a PyTorch Lightning trainer with the generation callback
        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path)
        resume = (
            os.path.join(checkpoint_path, "last.ckpt")
            if os.path.isfile(os.path.join(checkpoint_path, "last.ckpt"))
            else False
        )
        trainer = pl.Trainer(
            default_root_dir=checkpoint_path,
            gpus=1 if str(device).startswith("cuda") else 0,
            max_epochs=60,
            gradient_clip_val=0.1,
            callbacks=[
                ModelCheckpoint(
                    mode="min",
                    monitor="val_contrastive_divergence",
                    dirpath=checkpoint_path,
                    save_last=True,
                ),
                GenerateCallback(
                    num_steps=8192,
                    every_n_epochs=5,
                    train_type=train_type,
                    decom_level=decom_level,
                ),
                SamplerCallback(every_n_epochs=5),
                OutlierCallback(),
                LearningRateMonitor("epoch"),
            ],
            progress_bar_refresh_rate=1,
            logger=logger,
        )
        fig = plt.figure()
        plt.plot(train_set.time_series)
        trainer.logger.experiment.add_figure("Initial Time Series", fig)

        # Check whether pretrained model exists.
        # If yes, load it and skip training
        pl.seed_everything(seed)
        model = DeepEnergyModel(**kwargs)
        if bool(resume):
            trainer.fit(
                model, train_loader, validation_loader, ckpt_path=resume
            )
        else:
            trainer.fit(model, train_loader, validation_loader)
        model = DeepEnergyModel.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )
        # No testing as we are more interested in other properties
        return model

    print("Preparations Done! Starting Training...")
    model = train_model(
        img_shape=train_set[0].shape,
        batch_size=train_loader.batch_size,
        lr=1e-4,
        beta1=0.0,
    )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script for EBM")

    parser.add_argument("proj_name", type=str, help="Project Name")
    parser.add_argument(
        "train_type",
        type=str,
        choices=["decomposed", "pure"],
        help="Train Type",
    )
    parser.add_argument(
        "window",
        type=int,
        help="Window Size",
    )
    parser.add_argument(
        "decom_level",
        type=int,
        help="Decomposition Level",
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the dataset for the trainer",
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to the checkpoint for the trainer",
    )

    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        type=int,
        default=128,
        help="Batchsize for training",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=42,
        help="Seed for the random number generator",
    )
    args = parser.parse_args()

    train_type = (
        TrainType.PURE
        if args.train_type.lower() == "pure"
        else TrainType.DECOMPOSED
    )
    train_loop(
        train_type,
        args.window,
        args.decom_level,
        args.dataset_path,
        args.checkpoint_path,
        args.proj_name,
        args.batch_size,
        args.seed,
    )
