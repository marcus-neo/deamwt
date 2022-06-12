"""Module containing the Multiple Training Function."""
import argparse
from typing import Union
import os
import time

from test_single import test_loop
from train import train_loop
from utils.train_type import TrainType


ALL_DATA_PATH = "datasets"
ALL_CHECKPOINTS_PATH = "checkpoints"


def train_all(
    train_type: TrainType,
    window: int,
    decom_level: Union[int, None],
    run_test: bool,
) -> None:
    """Train the model on all the data."""
    if train_type == TrainType.DECOMPOSED:
        train_type_str = "decomposed"
        data_path = ALL_DATA_PATH
        checkpoints_path = os.path.join(
            ALL_CHECKPOINTS_PATH,
            f"window_{window}",
            f"decom_{decom_level}",
            "decomposed",
        )
    elif train_type == TrainType.PURE:
        train_type_str = "pure"
        data_path = ALL_DATA_PATH
        checkpoints_path = os.path.join(
            ALL_CHECKPOINTS_PATH,
            f"window_{window}",
            "pure",
        )

    projects = os.listdir(data_path)
    for project in projects:
        print(f"Performing Training on Project: {project}")
        proj_data = os.path.join(data_path, project)
        proj_checkpoint = os.path.join(checkpoints_path, project)
        proj_name = project
        start_time = time.time()
        train_loop(
            train_type,
            window,
            decom_level,
            proj_data,
            proj_checkpoint,
            proj_name,
            batch_size=128,
            seed=42,
        )
        end_time = time.time()
        with open("time.txt", "a", encoding="utf-8") as f:
            f.write(
                f"{project},{window},{decom_level},{end_time - start_time}\n"
            )
        if run_test:
            if train_type == TrainType.DECOMPOSED:
                output_path = os.path.join(
                    "output_images",
                    f"window_{window}",
                    f"decom_{decom_level}",
                    train_type_str,
                    proj_name,
                )
            elif train_type == TrainType.PURE:
                output_path = os.path.join(
                    "output_images",
                    f"window_{window}",
                    train_type_str,
                    proj_name,
                )
            test_loop(
                train_type,
                window,
                decom_level,
                proj_data,
                proj_checkpoint,
                output_path,
                seed=42,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script for EBM")

    parser.add_argument("training_type", type=str, help="Pure/Decomposed")
    parser.add_argument("window", type=int, help="Window Size")
    parser.add_argument("decom_level", type=int, help="Decomposition Level")

    parser.add_argument(
        "--test",
        dest="test",
        default="0",
        choices=["1", "0"],
        help="Perform Test After Training",
    )
    args = parser.parse_args()

    if args.training_type.lower() == "decomposed":
        train_all(
            TrainType.DECOMPOSED,
            args.window,
            args.decom_level,
            bool(int(args.test)),
        )
    elif args.training_type.lower() == "pure":
        train_all(TrainType.PURE, args.window, None, bool(int(args.test)))
    elif args.training_type.lower() == "both":
        print("Training Pure, Non-decomposed Data")
        train_all(
            TrainType.PURE, args.window, args.decom_level, bool(int(args.test))
        )
        print("Training Wavelet Decomposed Data")
        train_all(
            TrainType.DECOMPOSED,
            args.window,
            args.decom_level,
            bool(int(args.test)),
        )
    else:
        raise ValueError(f"Invalid Training Type: {args.training_type}")
