"""Module to Initialise the Datasets."""
import json
import os
from typing import Union, Optional

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.preprocessing import db1_multires_analysis
from utils.train_type import TrainType
from utils.errors import WindowDecomMismatchError, checker_train_type_decom


class TrainTestData:

    """Reader class for the dataset."""

    def _get_vals(self, data_path: str, window: int):
        """Get the data from the given path."""
        file = os.listdir(data_path)[0]
        scaler = MinMaxScaler()
        values = []
        with open(os.path.join(data_path, file), "r", encoding="utf-8") as f:
            dataframe = pd.read_csv(
                f,
                delimiter=",",
            )
            df_values = dataframe["value"]
            df_list = df_values.values.reshape(-1, 1)
            df_list = scaler.fit_transform(df_list).flatten()
            rolling = list(pd.DataFrame(df_list).rolling(window=window))[
                window - 1 :
            ]
            values.extend(rolling)
        return values, dataframe, df_list

    def __init__(self, dataset_directory: str, window: int):
        """Initialize the dataset."""
        train_path = os.path.join(dataset_directory, "train")
        train_val, _, self.train_df_list = self._get_vals(train_path, window)
        test_path = os.path.join(dataset_directory, "test")
        self.test, test_df, self.test_df_list = self._get_vals(
            test_path, window
        )
        with open(
            os.path.join(dataset_directory, "anomalies.json"),
            encoding="utf-8",
        ) as a_file:
            anomaly_dates: list = json.load(a_file)
            index = test_df.index
            self.anomalies = []
            for anomaly in anomaly_dates:
                condition = test_df["timestamp"] == anomaly
                self.anomalies.append(
                    index[condition].to_list()[0] - window + 1
                )
        self.train = train_val[: int(0.8 * len(train_val))]
        self.val = train_val[int(0.8 * len(train_val)) :]


class DatasetTrainVal(Dataset):

    """Dataset for Training and Validation Data."""

    def __init__(
        self,
        train_type: TrainType,
        df_list: np.ndarray,
        train_val: list,
        decom_level: Optional[Union[int, None]] = None,
    ):
        """Initialize the dataset."""
        self.train_type = train_type
        self.decom_level = decom_level
        if (
            self.train_type == TrainType.DECOMPOSED
            and self.decom_level is None
        ):
            raise ValueError("decom_level must be provided")
        if self.train_type == TrainType.PURE and self.decom_level is not None:
            raise ValueError("decom_level must be None")
        self.time_series = df_list
        self.train_val = train_val

        # Perform sanity check
        if self.train_type == TrainType.DECOMPOSED:
            print(
                "Multi-level decomposition selected.\n"
                "Performing sanity check of dataset."
            )
            sample: pd.DataFrame = self.train_val[0]
            try:
                db1_multires_analysis(
                    sample.to_numpy().flatten(), level=decom_level
                )
            except UserWarning as e:
                raise WindowDecomMismatchError(
                    "Window size and decom_level do not match"
                ) from e

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.train_val)

    def __getitem__(self, idx: Union[torch.Tensor, int]):
        """Obtain a sample from the dataset."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        item: pd.DataFrame = self.train_val[idx]
        if self.train_type == TrainType.DECOMPOSED:
            img = db1_multires_analysis(
                item.to_numpy().flatten(), level=self.decom_level
            )
            return np.expand_dims(img, 0)
        if self.train_type == TrainType.PURE:
            return np.expand_dims(item.to_numpy().flatten(), [0, -1])

        raise ValueError("TrainType not supported")


class DatasetTest(Dataset):

    """Dataset for Test Data."""

    def __init__(
        self,
        train_type: TrainType,
        test_val: list,
        test_df_list: np.ndarray,
        decom_level: Optional[Union[int, None]] = None,
    ):
        """Initialize the dataset."""
        self.train_type = train_type
        self.decom_level = decom_level
        checker_train_type_decom(self.decom_level, self.train_type)

        self.test_val = test_val
        self.df_list = test_df_list

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.test_val)

    def __getitem__(self, idx: Union[torch.Tensor, int]):
        """Get item from the dataset."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        item: pd.DataFrame = self.test_val[idx]
        if self.train_type == TrainType.DECOMPOSED:
            img = db1_multires_analysis(
                item.to_numpy().flatten(), level=self.decom_level
            )
            return np.expand_dims(img, 0)
        if self.train_type == TrainType.PURE:
            return np.expand_dims(item.to_numpy().flatten(), [0, -1])

        raise ValueError("TrainType not supported")
