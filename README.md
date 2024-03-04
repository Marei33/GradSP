# Gradient-based Smoothing Parameter Estimation for Neural P-splines

## Introduction

This repository contains the implementation of gradient-based smoothing parameter estimation for neural P-splines. The methodology facilitates the estimation of smoothing parameters via gradient-based optimization, specifically using the Adam optimizer, for neural network-based additive models.

## Features

- Implementation of neural P-splines.
- Gradient-based optimization of smoothing parameters using Generalized Cross Validation (GCV) and Restricted Maximum Likelihood (REML).
- Simulations and applications demonstrating the effectiveness and robustness of the approach.

## Requirements

- Python 3.10.9
- TensorFlow 2.11.0
- NumPy
- Other dependencies listed in `requirements.txt`


## Usage

- Demonstration of the fitting process of neural P-splines and the gradient-based smoothing parameter selection with the same initial starting point for the smoothing parameter can be found in `Optimization_SameStart.py` while in `Optimization_GridSearch.py` the starting point for the smoothing parameter is estimated via a small grid search.

- In `Optimization_OnlySpline.py` the optimization is implemented where the smoothing parameter is fixed and only the regression weights are optimized.

- Refer to the `multidimensional` directory for notebooks showing the optimization process fitting two P-splines.

- The directory `big-data` contains the optimization of a neural P-spline when having a huge dataset where batching is additionally needed.

