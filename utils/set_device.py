"""Module to set the pytorch device."""
import torch


def set_device():
    """Set the device to use, either GPU Cuda or CPU."""
    device = (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("Device:", device)
    return device
