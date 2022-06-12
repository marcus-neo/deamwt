"""Module containing the Test Loop Function."""
import argparse
import json
import os
from typing import Optional, List, Tuple


import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils import data
import pytorch_lightning as pl


# Imports for plotting
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from utils.anomaly_threshold import anomaly_threshold
from utils.train_type import TrainType
from utils.ebm import DeepEnergyModel
from utils.callbacks import GenerateCallback
from utils.datasets import TrainTestData, DatasetTrainVal, DatasetTest
from utils.set_device import set_device


matplotlib.rcParams["lines.linewidth"] = 2.0
sns.reset_orig()


# pylint: disable=too-many-statements,too-many-locals, redefined-outer-name
def test_loop(
    train_type: TrainType,
    window: int,
    decom_level: int,
    dataset_path: str,
    checkpoint_path: str,
    output_path: str,
    seed: Optional[int] = 42,
) -> Tuple[List[float], List[int]]:
    """Test the model on the test set."""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    all_data = TrainTestData(dataset_path, window)
    train_set = DatasetTrainVal(
        train_type=train_type,
        df_list=all_data.train_df_list,
        train_val=all_data.train,
        decom_level=decom_level,
    )
    test_set = DatasetTest(
        train_type=train_type,
        test_val=all_data.test,
        test_df_list=all_data.test_df_list,
        decom_level=decom_level,
    )
    # Path to the folder where the pretrained models are saved
    print("Initializing Test Script...")
    device = set_device()
    pretrained_filename = os.path.join(checkpoint_path, "last.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = DeepEnergyModel.load_from_checkpoint(pretrained_filename)
    else:
        raise FileNotFoundError(
            f"Could not find pretrained model at {pretrained_filename}."
        )
    model.to(device)
    print(f"Setting seed to {seed}")
    pl.seed_everything(seed)
    callback = GenerateCallback(
        batch_size=4,
        vis_steps=8,
        num_steps=4096,
        train_type=train_type,
        decom_level=decom_level,
    )
    imgs_per_step = callback.generate_fakes(model)
    imgs_per_step = imgs_per_step.cpu()
    # Signal Generation Phase
    # Check if signal generation folder exists, if not create it
    signal_generation_path = os.path.join(output_path, "signal_generation")
    if not os.path.exists(signal_generation_path):
        os.makedirs(signal_generation_path)
    # Save the generated images
    for i in range(imgs_per_step.size(1)):
        print(f"Performing Generation for Sample {i}")
        step_size = callback.num_steps // callback.vis_steps
        imgs_to_plot = imgs_per_step[step_size - 1 :: step_size, i]
        imgs_to_plot = torch.cat([imgs_per_step[0:1, i], imgs_to_plot], dim=0)
        for j in tqdm(range(imgs_to_plot.shape[0])):
            img = torch.squeeze(imgs_to_plot[j])
            if train_type == TrainType.DECOMPOSED:
                out = torch.sum(img, 1).numpy()
            elif train_type == TrainType.PURE:
                out = img.numpy()
            plt.plot(out)
            plt.savefig(
                os.path.join(signal_generation_path, f"sample_{i}_gen_{j}.png")
            )
        plt.close()

    # Perform Anomaly Detection
    print("Performing Anomaly Detection Phase")

    # Lower output scores mean lower probability => Low Confidence

    print("Phase 1: Detecting Random Noise")
    with torch.no_grad():
        rand_signals = torch.rand((128,) + model.hparams.img_shape).to(
            model.device
        )
        rand_signals = rand_signals * 2 - 1.0
        rand_out = model.cnn(rand_signals).mean()
        print(f"Average Score for Random Noise: {rand_out.item():4.2f}")

    print("Phase 2: Detecting Training Dataset")
    train_loader = data.DataLoader(
        train_set,
        batch_size=128,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = data.DataLoader(
        test_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4
    )
    with torch.no_grad():
        train_sigs = next(iter(train_loader))
        train_sigs = train_sigs.to(model.device)
        train_out = model.cnn(train_sigs).mean()
        print(f"Average Score for Training Dataset: {train_out:4.2f}")

    print("Phase 3: Detecting Test Dataset")

    out_list = []
    with torch.no_grad():
        for batch in iter(test_loader):
            for sig in batch:
                sig = torch.unsqueeze(sig, 0)
                sig = sig.to(model.device)
                out = model.cnn(sig).cpu().detach().numpy()
                out = out.reshape(-1)
                out_list.append(out)
        np.savetxt(
            os.path.join(output_path, "test_out.csv"), out_list, delimiter=","
        )
        _, axs = plt.subplots(2, 1)
        list_len_diff = len(test_set.df_list) - len(out_list)
        axs[0].plot(test_set.df_list[list_len_diff:])
        axs[1].plot(out_list)

        plt.savefig(os.path.join(output_path, "test_out.png"))

    return out_list, all_data.anomalies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Script for EBM")
    parser.add_argument(
        "name",
        type=str,
        help="Name of the test",
    )
    parser.add_argument(
        "training_type",
        type=str,
        choices=["decomposed", "pure"],
        help="The type of test to perform",
    )
    parser.add_argument(
        "window",
        type=int,
        help="The window size to use for test",
    )
    parser.add_argument(
        "decom_level",
        type=int,
        help="The decomposition level to use for test",
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the dataset for conducting the test",
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to the checkpoint for conducting the test",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to the output visualisation images",
        default="output_images/default",
    )

    # Path to the folder where the datasets are/should be downloaded
    # (e.g. CIFAR10)
    args = parser.parse_args()
    training_type = args.training_type
    output_path = args.output_path
    window = args.window
    decom_level = args.decom_level
    if training_type.lower() == "decomposed":
        out_list, anomaly_list = test_loop(
            TrainType.DECOMPOSED,
            window,
            decom_level,
            args.dataset_path,
            args.checkpoint_path,
            output_path,
        )
    elif training_type.lower() == "pure":
        out_list, anomaly_list = test_loop(
            TrainType.PURE,
            window,
            decom_level,
            args.dataset_path,
            args.checkpoint_path,
            output_path,
        )
    plots_path = os.path.join("test_outputs", "plots", args.name)
    vals_path = os.path.join("test_outputs", "vals", args.name)
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    if not os.path.exists(vals_path):
        os.makedirs(vals_path)

    # Plot Raw Signal
    for raw_data in os.listdir(os.path.join(args.dataset_path, "test")):
        raw_df = pd.read_csv(os.path.join(args.dataset_path, "test", raw_data))
    with open(
        os.path.join(args.dataset_path, "anomalies.json"), encoding="utf-8"
    ) as f:
        anomaly_dates = json.load(f)
        index = raw_df.index
        raw_anomalies = []
        for anomaly in anomaly_dates:
            condition = raw_df["timestamp"] == anomaly
            raw_anomalies.append(index[condition].to_list()[0])
    plt.figure(figsize=(6, 3.8))
    plt.plot(raw_df["value"].values)
    for anomaly in raw_anomalies:
        plt.axvline(anomaly, color="red", linestyle="--", linewidth="1")
    plt.title(f"{args.name}\nRaw Signal Input")
    plt.xlabel("Time Index")
    plt.ylabel("Signal Magnitude")
    plt.savefig(os.path.join(plots_path, "raw_signal.svg"))
    plt.close()

    plt.figure(figsize=(6, 3.8))
    threshold = anomaly_threshold(out_list)
    plt.plot(out_list)
    plt.xlabel("Time Index")
    plt.ylabel("Pred. Mag.")
    plt.title(f"Predictions on {args.name}")
    for coords in anomaly_list:
        plt.axvline(coords, color="red", linestyle="--", linewidth="1")
    plt.axhline(threshold, color="blue", linestyle="--", linewidth="1")
    plt.savefig(os.path.join(plots_path, "predictions.svg"))
    plt.close()
