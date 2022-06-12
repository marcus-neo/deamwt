"""Pytest for the datasets module."""
import numpy as np
import pytest

from utils.preprocessing import db1_multires_analysis


class TestClass:

    """Test set for the db1_multires_analysis function."""

    def test_level_one_shape(self):
        """Test that the shape of the decomposition is correct at level 1."""
        rand_sig = np.random.rand(50)
        decomposed = db1_multires_analysis(rand_sig, 1)
        assert decomposed.shape == (50, 2)

    def test_level_two_shape(self):
        """Test that the shape of the decomposition is correct at level 2."""
        rand_sig = np.random.rand(50)
        decomposed = db1_multires_analysis(rand_sig, 2)
        assert decomposed.shape == (50, 3)

    def test_level_three_shape(self):
        """Test that the shape of the decomposition is correct at level 3."""
        rand_sig = np.random.rand(50)
        decomposed = db1_multires_analysis(rand_sig, 3)
        assert decomposed.shape == (50, 4)

    def test_level_four_shape(self):
        """Test that the shape of the decomposition is correct at level 4."""
        rand_sig = np.random.rand(50)
        decomposed = db1_multires_analysis(rand_sig, 4)
        assert decomposed.shape == (50, 5)

    def test_level_five_shape(self):
        """Test that the shape of the decomposition is correct at level 5."""
        rand_sig = np.random.rand(50)
        decomposed = db1_multires_analysis(rand_sig, 5)
        assert decomposed.shape == (50, 6)

    def test_level_one_sum(self):
        """Test that the sum of the decomposition is equal to the original."""
        rand_sig = np.random.rand(50)
        decomposed = db1_multires_analysis(rand_sig, 1)
        out = np.sum(decomposed, axis=1)
        assert np.allclose(out, rand_sig)

    def test_level_two_sum(self):
        """Test that the sum of the decomposition is equal to the original."""
        rand_sig = np.random.rand(50)
        decomposed = db1_multires_analysis(rand_sig, 2)
        out = np.sum(decomposed, axis=1)
        assert np.allclose(out, rand_sig)

    def test_level_three_sum(self):
        """Test that the sum of the decomposition is equal to the original."""
        rand_sig = np.random.rand(50)
        decomposed = db1_multires_analysis(rand_sig, 3)
        out = np.sum(decomposed, axis=1)
        assert np.allclose(out, rand_sig)

    def test_level_four_sum(self):
        """Test that the sum of the decomposition is equal to the original."""
        rand_sig = np.random.rand(50)
        decomposed = db1_multires_analysis(rand_sig, 4)
        out = np.sum(decomposed, axis=1)
        assert np.allclose(out, rand_sig)

    def test_level_five_sum(self):
        """Test that the sum of the decomposition is equal to the original."""
        rand_sig = np.random.rand(50)
        decomposed = db1_multires_analysis(rand_sig, 5)
        out = np.sum(decomposed, axis=1)
        assert np.allclose(out, rand_sig)

    def test_raise_window_decom_error_one(self):
        """Window Decom Error Test 2.

        Test that the function raises an error when the window size is
        smaller than pow(decomposition level, 2).
        """
        with pytest.warns(UserWarning):
            rand_sig = np.random.rand(1)
            db1_multires_analysis(rand_sig, 1)

    def test_raise_window_decom_error_two(self):
        """Window Decom Error Test 2.

        Test that the function raises an error when the window size is
        smaller than pow(decomposition level, 2).
        """
        with pytest.warns(UserWarning):
            rand_sig = np.random.rand(3)
            db1_multires_analysis(rand_sig, 2)

    def test_raise_window_decom_error_three(self):
        """Window Decom Error Test 3.

        Test that the function raises an error when the window size is
        smaller than pow(decomposition level, 2).
        """
        with pytest.warns(UserWarning):
            rand_sig = np.random.rand(7)
            db1_multires_analysis(rand_sig, 3)

    def test_raise_window_decom_error_four(self):
        """Window Decom Error Test 4.

        Test that the function raises an error when the window size is
        smaller than pow(decomposition level, 2).
        """
        with pytest.warns(UserWarning):
            rand_sig = np.random.rand(15)
            db1_multires_analysis(rand_sig, 4)

    def test_raise_window_decom_error_five(self):
        """Window Decom Error Test 5.

        Test that the function raises an error when the window size is
        smaller than pow(decomposition level, 2).
        """
        with pytest.warns(UserWarning):
            rand_sig = np.random.rand(31)
            db1_multires_analysis(rand_sig, 5)
