"""Module containing the error classes and error checkers."""

from utils.train_type import TrainType


def checker_train_type_decom(decom_level: int, train_type: TrainType):
    """Checker Function for Potential TrainType and Decom Mismatch."""
    if train_type == TrainType.DECOMPOSED and decom_level is None:
        raise ValueError("decom_level must be provided")
    if train_type == TrainType.PURE and decom_level is not None:
        raise ValueError("decom_level must be None")


class WindowDecomMismatchError(Exception):

    """Window and Decom Level Mismatch Error Class."""

    def __init__(self, message=None):
        """Initialise the Class."""
        super().__init__()
        if message is None:
            message = "Window size and decom_level do not match"
        self.message = message

    def __str__(self):
        """Return the message."""
        return repr(self.message)


class InternalModelError(Exception):

    """Internal Neural Network Error Class."""

    def __init__(self, message=None):
        """Initialise the Class."""
        super().__init__()
        if message is None:
            message = "Sanity Check not Passed, "
            "Check the Internal Neural Network"
        self.message = message

    def __str__(self):
        """Return the message."""
        return repr(self.message)
