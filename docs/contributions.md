# Contributing to deaMWT

👋 Hi Developers! Thank you so much for contributing to deaMWT!

This file serves as a guide for you to contribute your ideas onto the project.

## Table of Contents

[What are the Stages of deaMWT?](#components)

[How do I Get Started on Contributing?](#contributing)

[How do I Handle Errors while Contributing?](#errors)

## What are the Components of deaMWT? <a name="components"></a>

The two main stages of deaMWT are the **Data Preprocessing** and the **Energy Based Model**.

The **Data Preprocessing** stage refers to the decomposition of input signals using multiresolution wavelets decomposition.

The **Energy Based Model** stage refers to the training and testing of the decomposed signal with the energy based model.

## How do I Get Started on Contributing? <a name="contributing"></a>

To begin contributing to deaMWT, first fork the repository, make your changes, then submit a pull request into the project's `develop` branch.

Here are some suggestions for you to begin contributing to deaMWT. However, you may contribute to it in any way, so do not be restricted by this list!

### <ins> Contributing to Data Preprocessing Stage </ins>

**Introducing Different Wavelet Families**
Currently the project only support db1 wavelet from the Daubechies family of wavelets as its mother wavelet. Feel free to include other wavelets into the program!
Changes in this aspect can be done in the [preprocessing.py](../utils/preprocessing.py) script.

### <ins> Contributing to the Energy Based Model </ins>

**Adding a New Custom Dataset**
For the modular dataset loader to work, the custom dataset must be a folder with a specific folder structure. This dataset folder must reside in the [datasets](../datasets) folder. It must contain a train and test folder, and within each folder, a text file containing the indexed time series data must exist. There must also be file containing the indices of the anomalies, titled `anomalies.json`.
```
dataset_name
│   anomalies.json
│
└───train
│   └───train_0.txt
│
└───test
    └───test_0.txt
```
Within the train and test text files, the structure of the data should be as follows:
```
timestamp, value
2014-05-14 01:14:00,85.835
2014-05-14 01:19:00,88.167
2014-05-14 01:24:00,44.595
2014-05-14 01:29:00,56.282
2014-05-14 01:34:00,36.534
2014-05-14 01:39:00,36.894
...
...
```
The heading should be "timestamp, value". However, within the timestamp column, the values need not be timestamps, and can also be time indices. 
**Changing Internal Convolutional Neural Network**
Currently the internal convolutional neural network has only 3 Conv2D layers. This model is therefore fairly simplified, and perhaps more robust models can be used in its place.
Do note that the model needs to dynamically accept data with different sliding window lengths (it is safe to assume that the lengths do not change between datasets).
Changes in this area can be done in the [model.py](../utils/model.py) script.

**Changing the Type of Internal Neural Network**
Perhaps you might think that a CNN is not ideal in this aspect, and maybe LSTM models would be better here. If thats the case, feel free to replace the internal CNN with that of an LSTM model!
Changes in this area can be done in both the [model.py](../utils/model.py) and [ebm.py](../utils/ebm.py) scripts.

**Changing the way Anomalies are Created**
The current means of data preprocessing is unrealistic. It normalises
the entire dataset first, before applying the sliding window onto the normalised data. However, in production, this is not optimal because streaming data is constantly being received by the energy-based model. Repeatedly normalising the entire dataset at each time interval is time consuming and computationally sub-optimal. An alternate solution is to first identify a large enough sample with no anomalies, then set its absolute minimum as -1, and its absolute maximum as 1. All new windowed data will then be scaled with respect to this maximum and minimum. As a result, anomalies should always have a value that is much larger than 1 or much smaller than -1.

Changes to this portion of the code can be done in the [datasets.py](../utils/datasets.py).
