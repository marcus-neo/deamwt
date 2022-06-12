"""Batch Script for Multiple Tests."""
import os
import json
import numpy as np
import pandas as pd

# Imports for plotting
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from test_single import test_loop
from utils.anomaly_threshold import anomaly_threshold
from utils.train_type import TrainType


matplotlib.rcParams["lines.linewidth"] = 2.0
matplotlib.rcParams["font.family"] = "serif"
plt.rcParams["font.sans-serif"] = ["Times New Roman"]
sns.reset_orig()


class SpecificTest:

    """A class for testing specific models."""

    def __init__(self, name):
        """Initialize the class."""
        self.name = name
        self.args = [
            (16, 4),
            (32, 4),
            (64, 4),
            (128, 4),
            (32, 1),
            (32, 2),
            (32, 3),
            (32, 5),
            (32, None),
            (16, 2),
            (64, 2),
            (128, 2),
        ]
        self.paths = [
            os.path.join(
                "checkpoints",
                "window_16",
                "decom_4",
                "decomposed",
                self.name,
            ),
            os.path.join(
                "checkpoints",
                "window_32",
                "decom_4",
                "decomposed",
                self.name,
            ),
            os.path.join(
                "checkpoints",
                "window_64",
                "decom_4",
                "decomposed",
                self.name,
            ),
            os.path.join(
                "checkpoints",
                "window_128",
                "decom_4",
                "decomposed",
                self.name,
            ),
            os.path.join(
                "checkpoints",
                "window_32",
                "decom_1",
                "decomposed",
                self.name,
            ),
            os.path.join(
                "checkpoints",
                "window_32",
                "decom_2",
                "decomposed",
                self.name,
            ),
            os.path.join(
                "checkpoints",
                "window_32",
                "decom_3",
                "decomposed",
                self.name,
            ),
            os.path.join(
                "checkpoints",
                "window_32",
                "decom_5",
                "decomposed",
                self.name,
            ),
            os.path.join("checkpoints", "window_32", "pure", self.name),
            os.path.join(
                "checkpoints",
                "window_16",
                "decom_2",
                "decomposed",
                self.name,
            ),
            os.path.join(
                "checkpoints",
                "window_64",
                "decom_2",
                "decomposed",
                self.name,
            ),
            os.path.join(
                "checkpoints",
                "window_128",
                "decom_2",
                "decomposed",
                self.name,
            ),
        ]


# pylint: disable=too-many-statements
def test_all():
    """Test all Datasets."""
    test_names = [
        "ambient_temp_sys_fail",
        "cpu_misconfiguration",
        "ec2_request_latency_system_failure",
        "machine_temp_sys_fail",
        "nyc_taxi",
        "rogue_agent_key_hold",
        "synthetic",
    ]
    for name in test_names:
        test = SpecificTest(name)
        subplots = []
        anomalies = []
        plots_path = os.path.join("test_outputs", "plots", name)
        vals_path = os.path.join("test_outputs", "vals", name)
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)
        if not os.path.exists(vals_path):
            os.makedirs(vals_path)

        # Plot Raw Signal
        for raw_data in os.listdir(os.path.join("datasets", name, "test")):
            raw_df = pd.read_csv(
                os.path.join("datasets", name, "test", raw_data)
            )
        with open(
            os.path.join("datasets", name, "anomalies.json"), encoding="utf-8"
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
        plt.title(f"{name}\nRaw Signal Input")
        plt.xlabel("Time Index")
        plt.ylabel("Signal Magnitude")
        plt.savefig(os.path.join(plots_path, "raw_signal.svg"))
        plt.close()
        if not os.listdir(vals_path):
            for args, path in zip(test.args, test.paths):
                if args[1] is None:
                    train_type = TrainType.PURE
                elif args[1] is not None:
                    train_type = TrainType.DECOMPOSED

                out_list, anomaly_list = test_loop(
                    train_type,
                    args[0],
                    args[1],
                    os.path.join("datasets", name),
                    path,
                    path.replace("checkpoints", "output_images"),
                )
                subplots.append(out_list)
                anomalies.append(anomaly_list)
            np.save(os.path.join(vals_path, "subplots"), subplots)
            np.save(os.path.join(vals_path, "anomalies"), anomalies)
        subplots = np.load(
            os.path.join(vals_path, "subplots.npy"), allow_pickle=True
        )
        anomalies = np.load(
            os.path.join(vals_path, "anomalies.npy"), allow_pickle=True
        )

        # Figure for w32D0-w32D5
        _, axs = plt.subplots(3, 2, sharey=True)
        axs[0, 0].plot(subplots[8])
        axs[0, 0].set_xlabel("Time Index")
        axs[0, 0].set_ylabel("Pred. Mag.")
        threshold = anomaly_threshold(subplots[8])
        axs[0, 0].axhline(
            y=threshold, color="b", linestyle="--", linewidth="1"
        )
        for coords in anomalies[8]:
            axs[0, 0].axvline(
                x=coords, color="r", linestyle="--", linewidth="1"
            )
        axs[0, 0].set_title("No Decomposition")

        axs[0, 1].plot(subplots[4])
        axs[0, 1].set_xlabel("Time Index")
        axs[0, 1].set_ylabel("Pred. Mag.")
        threshold = anomaly_threshold(subplots[4])
        axs[0, 1].axhline(
            y=threshold, color="b", linestyle="--", linewidth="1"
        )
        for coords in anomalies[4]:
            axs[0, 1].axvline(
                x=coords, color="r", linestyle="--", linewidth="1"
            )
        axs[0, 1].set_title("Level 1 Decomposition")

        axs[1, 0].plot(subplots[5])
        axs[1, 0].set_xlabel("Time Index")
        axs[1, 0].set_ylabel("Pred. Mag.")
        threshold = anomaly_threshold(subplots[5])
        axs[1, 0].axhline(
            y=threshold, color="b", linestyle="--", linewidth="1"
        )
        for coords in anomalies[5]:
            axs[1, 0].axvline(
                x=coords, color="r", linestyle="--", linewidth="1"
            )
        axs[1, 0].set_title("Level 2 Decomposition")

        axs[1, 1].plot(subplots[6])
        axs[1, 1].set_xlabel("Time Index")
        axs[1, 1].set_ylabel("Pred. Mag.")
        threshold = anomaly_threshold(subplots[6])
        axs[1, 1].axhline(
            y=threshold, color="b", linestyle="--", linewidth="1"
        )
        for coords in anomalies[6]:
            axs[1, 1].axvline(
                x=coords, color="r", linestyle="--", linewidth="1"
            )
        axs[1, 1].set_title("Level 3 Decomposition")

        axs[2, 0].plot(subplots[1])
        axs[2, 0].set_xlabel("Time Index")
        axs[2, 0].set_ylabel("Pred. Mag.")
        threshold = anomaly_threshold(subplots[1])
        axs[2, 0].axhline(
            y=threshold, color="b", linestyle="--", linewidth="1"
        )
        for coords in anomalies[1]:
            axs[2, 0].axvline(
                x=coords, color="r", linestyle="--", linewidth="1"
            )
        axs[2, 0].set_title("Level 4 Decomposition")

        axs[2, 1].plot(subplots[7])
        axs[2, 1].set_xlabel("Time Index")
        axs[2, 1].set_ylabel("Pred. Mag.")
        threshold = anomaly_threshold(subplots[7])
        axs[2, 1].axhline(
            y=threshold, color="b", linestyle="--", linewidth="1"
        )
        for coords in anomalies[7]:
            axs[2, 1].axvline(
                x=coords, color="r", linestyle="--", linewidth="1"
            )
        axs[2, 1].set_title("Level 5 Decomposition")
        plt.suptitle(f"{name}\nVariable Decomposition Levels")
        plt.tight_layout()
        plt.subplots_adjust(top=0.8)
        plt.savefig(os.path.join(plots_path, "w32D0-w32D5.svg"))
        plt.close()

        # Figure for w16D4-w128D4
        _, axs = plt.subplots(2, 2, sharey=True)
        axs[0, 0].plot(subplots[0])
        axs[0, 0].set_xlabel("Time Index")
        axs[0, 0].set_ylabel("Pred. Mag.")
        threshold = anomaly_threshold(subplots[0])
        axs[0, 0].axhline(
            y=threshold, color="b", linestyle="--", linewidth="1"
        )
        for coords in anomalies[0]:
            axs[0, 0].axvline(
                x=coords, color="r", linestyle="--", linewidth="1"
            )
        axs[0, 0].set_title("Window Size 16")

        axs[0, 1].plot(subplots[1])
        axs[0, 1].set_xlabel("Time Index")
        axs[0, 1].set_ylabel("Pred. Mag.")
        threshold = anomaly_threshold(subplots[1])
        axs[0, 1].axhline(
            y=threshold, color="b", linestyle="--", linewidth="1"
        )
        for coords in anomalies[1]:
            axs[0, 1].axvline(
                x=coords, color="r", linestyle="--", linewidth="1"
            )
        axs[0, 1].set_title("Window Size 32")

        axs[1, 0].plot(subplots[2])
        axs[1, 0].set_xlabel("Time Index")
        axs[1, 0].set_ylabel("Pred. Mag.")
        threshold = anomaly_threshold(subplots[2])
        axs[1, 0].axhline(
            y=threshold, color="b", linestyle="--", linewidth="1"
        )
        for coords in anomalies[2]:
            axs[1, 0].axvline(
                x=coords, color="r", linestyle="--", linewidth="1"
            )
        axs[1, 0].set_title("Window Size 64")

        axs[1, 1].plot(subplots[3])
        axs[1, 1].set_xlabel("Time Index")
        axs[1, 1].set_ylabel("Pred. Mag.")
        threshold = anomaly_threshold(subplots[3])
        axs[1, 1].axhline(
            y=threshold, color="b", linestyle="--", linewidth="1"
        )
        for coords in anomalies[3]:
            axs[1, 1].axvline(
                x=coords, color="r", linestyle="--", linewidth="1"
            )
        axs[1, 1].set_title("Window Size 128")
        plt.suptitle(f"{name}\nVariable Window Sizes, Decom Level 4")
        plt.tight_layout()
        plt.subplots_adjust(top=0.8)
        plt.savefig(os.path.join(plots_path, "w16D4-w128D4.svg"))
        plt.close()
        # Figure for w16D4-w128D4
        _, axs = plt.subplots(2, 2)
        axs[0, 0].plot(subplots[9])
        axs[0, 0].set_xlabel("Time Index")
        axs[0, 0].set_ylabel("Pred. Mag.")
        threshold = anomaly_threshold(subplots[9])
        axs[0, 0].axhline(
            y=threshold, color="b", linestyle="--", linewidth="1"
        )
        for coords in anomalies[9]:
            axs[0, 0].axvline(
                x=coords, color="r", linestyle="--", linewidth="1"
            )
        axs[0, 0].set_title("Window Size 16")
        axs[0, 1].plot(subplots[5])
        axs[0, 1].set_xlabel("Time Index")
        axs[0, 1].set_ylabel("Pred. Mag.")
        threshold = anomaly_threshold(subplots[5])
        axs[0, 1].axhline(
            y=threshold, color="b", linestyle="--", linewidth="1"
        )
        for coords in anomalies[5]:
            axs[0, 1].axvline(
                x=coords, color="r", linestyle="--", linewidth="1"
            )
        axs[0, 1].set_title("Window Size 32")
        axs[1, 0].plot(subplots[10])
        axs[1, 0].set_xlabel("Time Index")
        axs[1, 0].set_ylabel("Pred. Mag.")
        threshold = anomaly_threshold(subplots[10])
        axs[1, 0].axhline(
            y=threshold, color="b", linestyle="--", linewidth="1"
        )
        for coords in anomalies[10]:
            axs[1, 0].axvline(
                x=coords, color="r", linestyle="--", linewidth="1"
            )
        axs[1, 0].set_title("Window Size 64")
        axs[1, 1].plot(subplots[11])
        axs[1, 1].set_xlabel("Time Index")
        axs[1, 1].set_ylabel("Pred. Mag.")
        threshold = anomaly_threshold(subplots[11])
        axs[1, 1].axhline(
            y=threshold, color="b", linestyle="--", linewidth="1"
        )
        for coords in anomalies[11]:
            axs[1, 1].axvline(
                x=coords, color="r", linestyle="--", linewidth="1"
            )
        axs[1, 1].set_title("Window Size 128")
        plt.suptitle(f"{name}\nVariable Window Sizes, Decom Level 2")
        plt.tight_layout()
        plt.subplots_adjust(top=0.8)
        plt.savefig(os.path.join(plots_path, "w16D2-w128D2.svg"))
        plt.close()


if __name__ == "__main__":
    test_all()
