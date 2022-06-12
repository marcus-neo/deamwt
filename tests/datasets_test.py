"""Pytest for the datasets module."""

import os

import pandas as pd
import pytest

from utils.train_type import TrainType
from utils.datasets import TrainTestData, DatasetTrainVal, DatasetTest


class TestTrainTestData:

    """Test set for the TrainTestData class."""

    def test_load_file(self):
        """Test if files can be successfully loaded."""
        data_path = os.path.join("datasets", "nyc_taxi")
        _ = TrainTestData(data_path, window=30)

    def test_anomalies_outer_type(self):
        """Test Anomalies Outer Type."""
        data_path = os.path.join("datasets", "nyc_taxi")
        test_class = TrainTestData(data_path, window=30)
        anomalies = test_class.anomalies
        assert isinstance(anomalies, list)

    def test_anomalies_inner_type(self):
        """Test Anomalies inner Type."""
        data_path = os.path.join("datasets", "nyc_taxi")
        test_class = TrainTestData(data_path, window=30)
        anomalies = test_class.anomalies
        assert isinstance(anomalies[0], int)

    def test_anomalies_length(self):
        """Test Anomalies Length."""
        data_path = os.path.join("datasets", "nyc_taxi")
        test_class = TrainTestData(data_path, window=30)
        anomalies = test_class.anomalies
        assert len(anomalies) == 5, "anomalies mismatch"

    def test_train_outer_type(self):
        """Test the Training Outer Type."""
        data_path = os.path.join("datasets", "nyc_taxi")
        test_class = TrainTestData(data_path, window=30)
        train = test_class.train
        assert isinstance(train, list)

    def test_train_inner_type(self):
        """Test the Training Inner Type."""
        data_path = os.path.join("datasets", "nyc_taxi")
        test_class = TrainTestData(data_path, window=30)
        train = test_class.train
        assert isinstance(train[0], pd.DataFrame)

    def test_train_length(self):
        """Test the Training Length."""
        data_path = os.path.join("datasets", "nyc_taxi")
        test_class = TrainTestData(data_path, window=30)
        train = test_class.train
        assert (
            len(train) == int(0.8 * (len(test_class.train_df_list) - 30)) + 1
        )

    def test_val_outer_type(self):
        """Test the Validation Outer Type."""
        data_path = os.path.join("datasets", "nyc_taxi")
        test_class = TrainTestData(data_path, window=30)
        val = test_class.val
        # Check if val is a list
        assert isinstance(val, list)

    def test_val_length(self):
        """Test the Validation Length."""
        data_path = os.path.join("datasets", "nyc_taxi")
        test_class = TrainTestData(data_path, window=30)
        val = test_class.val
        # Check if val is a list of length equal to the number of elements
        assert len(val) == int(0.2 * (len(test_class.train_df_list) - 30)) + 1

    def test_val_inner_type(self):
        """Test the Validation Inner Type."""
        data_path = os.path.join("datasets", "nyc_taxi")
        test_class = TrainTestData(data_path, window=30)
        val = test_class.val
        # Check if val is a list of floats
        assert isinstance(val[0], pd.DataFrame)

    def test_test_outer_type(self):
        """Test the Testing Outer Type."""
        data_path = os.path.join("datasets", "nyc_taxi")
        test_class = TrainTestData(data_path, window=30)
        test = test_class.test
        assert isinstance(test, list)

    def test_test_inner_type(self):
        """Test the Testing Inner Type."""
        data_path = os.path.join("datasets", "nyc_taxi")
        test_class = TrainTestData(data_path, window=30)
        test = test_class.test
        assert isinstance(test[0], pd.DataFrame)

    def test_test_length(self):
        """Test the Testing Length."""
        data_path = os.path.join("datasets", "nyc_taxi")
        test_class = TrainTestData(data_path, window=30)
        test = test_class.test
        assert len(test) == len(test_class.test_df_list) - 30 + 1


class TestDatasetTrainVal:

    """Test set for the DatasetTrainVal class."""

    def test_init_params_1(self):
        """Test the initialization of the class."""
        with pytest.raises(ValueError):
            data_path = os.path.join("datasets", "nyc_taxi")
            train_test_data = TrainTestData(data_path, window=30)
            _ = DatasetTrainVal(
                train_type=TrainType.DECOMPOSED,
                df_list=train_test_data.train_df_list,
                train_val=train_test_data.train,
                decom_level=None,
            )

    def test_init_params_2(self):
        """Test the initialization of the class."""
        with pytest.raises(ValueError):
            data_path = os.path.join("datasets", "nyc_taxi")
            train_test_data = TrainTestData(data_path, window=30)
            _ = DatasetTrainVal(
                train_type=TrainType.PURE,
                df_list=train_test_data.train_df_list,
                train_val=train_test_data.train,
                decom_level=1,
            )

    def test_init_user_warning(self):
        """Test the initialization of the class."""
        with pytest.warns(UserWarning):
            data_path = os.path.join("datasets", "nyc_taxi")
            train_test_data = TrainTestData(data_path, window=2)
            _ = DatasetTrainVal(
                train_type=TrainType.DECOMPOSED,
                df_list=train_test_data.train_df_list,
                train_val=train_test_data.train,
                decom_level=4,
            )

    def test_get_item_function(self):
        """Test the get item function."""
        data_path = os.path.join("datasets", "nyc_taxi")
        train_test_data = TrainTestData(data_path, window=30)
        train_val = DatasetTrainVal(
            train_type=TrainType.DECOMPOSED,
            df_list=train_test_data.train_df_list,
            train_val=train_test_data.train,
            decom_level=4,
        )
        assert train_val[0].shape == (1, 30, 5)


class TestDatasetTest:

    """Test set for the DatasetTest class."""

    def test_init_params_1(self):
        """Test the initialization of the class."""
        with pytest.raises(ValueError):
            data_path = os.path.join("datasets", "nyc_taxi")
            train_test_data = TrainTestData(data_path, window=30)
            _ = DatasetTest(
                train_type=TrainType.DECOMPOSED,
                test_df_list=train_test_data.train_df_list,
                test_val=train_test_data.test,
                decom_level=None,
            )

    def test_init_params_2(self):
        """Test the initialization of the class."""
        with pytest.raises(ValueError):
            data_path = os.path.join("datasets", "nyc_taxi")
            train_test_data = TrainTestData(data_path, window=30)
            _ = DatasetTest(
                train_type=TrainType.PURE,
                test_df_list=train_test_data.train_df_list,
                test_val=train_test_data.test,
                decom_level=1,
            )

    def test_get_item_function(self):
        """Test the get item function."""
        data_path = os.path.join("datasets", "nyc_taxi")
        train_test_data = TrainTestData(data_path, window=30)
        test_val = DatasetTest(
            train_type=TrainType.DECOMPOSED,
            test_df_list=train_test_data.test_df_list,
            test_val=train_test_data.train,
            decom_level=4,
        )
        assert test_val[0].shape == (1, 30, 5)
