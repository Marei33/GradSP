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

## Installation

1. Clone the repository: 'git clone https://github.com/Marei33/DDL-PSplines.git'

2. Install the required dependencies: 'pip install -r requirements.txt'


## Usage

- Demonstration of the fitting process of neural P-splines and the gradient-based smoothing parameter selection can be found in `opti.py` while in `opti_splines.py` the smoothing parameter is fixed and only the regression weights are optimized.

- Refer to the `multidimensional` directory for notebooks showing the optimization process fitting two P-splines.

- Directory `big-data` contains the optimization of a neural P-spline when having a huge dataset where batching is needed.

