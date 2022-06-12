"""Module for training the model back-to-back."""

from train_all import train_all
from utils.train_type import TrainType


def batch_script():
    """Script for training the model back-to-back."""
    combination_list = {
        "window": [16, 64, 128],
        "decom_level": [2, 2, 2],
    }
    for counter in range(len(combination_list["window"])):
        train_all(
            train_type=TrainType.DECOMPOSED,
            window=combination_list["window"][counter],
            decom_level=combination_list["decom_level"][counter],
            run_test=False,
        )


if __name__ == "__main__":
    batch_script()
