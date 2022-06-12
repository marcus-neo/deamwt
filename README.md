# Data Stream Evolution Analysis using Multiresolution Wavelet Transform

[![CICD](https://github.com/marcus-neo/deamwt/actions/workflows/master-workflow.yml/badge.svg)](https://github.com/marcus-neo/deamwt/actions/workflows/master-workflow.yml)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## deaMWT Abstract

Data-streams have become essential in today's information-driven world, and uncovering information from these data-streams have become increasingly vital for making predictions and detecting errors. With the increasing complexity of data-streams, traditional analysis methods are becoming obsolete, while modern-day machine learning models are constantly improving to handle these data-stream. Yet despite the exceptional performances that machine learning model boasts, information still cannot be completely extracted from raw data, and data pre-processing is required for clearer patterns to emerge.
This project identifies the ability of the multiresolution wavelet transform to decompose a stream of data into orthogonal bases, and hypothesises that this could be used as a pre-processing method to extract essential features of the data-stream before they are being used as inputs to the machine learning model. By utilising an energy-based model to detect anomalies in within several sophisticated datasets, this project aims to confirm the hypothesis that using the multiresolution wavelet transform as a feature extractor will improve the performance of machine learning models.

## Install

Using the command line, the project can be initialised by running the following code:

```
pip install -r requirements.txt
```

## Running Training and Tests

The `datasets` folder contains several dataset samples to select from.

### Running a Single Training and Test

To run training on a single dataset, input the following into the command line:

```
python train.py <PROJECT_NAME> decomposed <SLIDING_WINDOW_LENGTH> <DECOMPOSITION_LEVEL> datasets/<DATASET_NAME> checkpoints/<CHECKPOINT_NAME>
```

Upon completing the training, run the test with the following code:

```
python test_single.py <PROJECT_NAME> decomposed <SLIDING_WINDOW_LENGTH> <DECOMPOSITION LEVEL> datasets/<DATASET_NAME> checkpoints/<CHECKPOINT_NAME> <OUTPUT_PATH>
```

### Running a Sequence of Trainings and Tests

To sequentially run training and tests on every dataset within the `datasets` folder, use the `train_all.py` script as below:

```
python3 train_all.py --test 1 decomposed <SLIDING_WINDOW_LENGTH> <DECOMPOSITION_LEVEL>
```

To only run the sequence of training, and not the tests, set the `test` flag to 0.

```
python train_all.py --test 0 decomposed <SLIDING_WINDOW_LENGTH> <DECOMPOSITION_LEVEL>
```

### Running a Sequences of Trainings and Tests with Varying Hyperparameters

To sequentially run trainings and tests on every dataset within the `datasets` folder, while varying the hyperparameters, use the `training_hyperparam.py` script as below:

```
python train_hyperparam.py
```

NOTE: It is required to edit the `train_hyperparam.py` file to select the list of hyperparameters of your choice.

After completing the training, run the test script

```
python test_batch.py
```

NOTE: Once again, it is required to edit the `test_batch.py` file to select the list of hyperparameters of your choice.

After performing training and tests, the test results can be obtained from the folder `test_outputs`. Within this folder, the subfolder `plots` will contain the plots of the results, `signal_generation` will contain the fake data generated by the energy based model, and `vals` will contain the raw results.

## Visualisation on Tensorboard

To visiaulse the training progress in realtime, while the training is running, open a new command line and enter:

```
tensorbord --logdir logs
```

After which, go to the link http://localhost:6006

## Is there a More In-Depth Detail about this Project?

Yes! Take a look at the paper located [here](docs/paper.pdf)

## I would like to Contribute!

That is great! Please look at the [contributions](docs/contributions.md) file on how you can go about contributing to this project!

## Found this project useful, cite it!

If you have found this project useful, you can cite it by simply copying this biblatex code:

```biblatex
@mastersthesis{neo2022data,
  title={Data Stream Evolution Analysis using Multiresolution Wavelet Transform},
  author={Marcus J. Q. Neo},
  school={Imperial College London},
  year={2022}
}
```