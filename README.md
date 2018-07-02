# SC-DNN

## Overview
This repository provides source code and documentation for the implementation of Neural Networks using Stochastic Computing, a novel computing paradigm that provides significantly lower hardware footprint compared to conventional binary computing. A software implementation of several computational elements is provided including scaled addition, multiplication, scaled inner product and a stochastic implementation of the hyperbolic function. Furthermore, a stochastic comparator is implemented and used to introduce a stochastic implementation of the rectified linear unit (ReLU). Saturation arithmetic elements are also introduced based on a stochastic linear gain function. A modified neuron architecture is proposed and impolemented to model stochastic arithmetic during the training phase of a network.

## Setup 

### macOS

## User Guide

The script `scripts/load_mnist.py` loads the normalised MNIST dataset used for inference purposes. While being the `SC-DNN/` directory run 

    python scripts/load_mnist.py 

This saves the testing data features and labels in the `data/` directory. 

The directory `src/inference` contains the implementation of several SC processing elements in `src/inference/function_blocks.py` as well as the implementation of the scaling operations in `src/inference/scaling_ops.py` that are used to determine signal scalings in the SC network graph. The directories `src/inference/network_I` and `src/inference/network_II` contain trained coefficients for the two test-case networks studied in the [Report](https://github.com/adamosSol/SC-DNN/blob/master/docs/Report.pdf). 

## Documentation
[Report](https://github.com/adamosSol/SC-DNN/blob/master/docs/Report.pdf): Analysis and evaluation  

[Resources](https://github.com/adamosSol/SC-DNN/blob/master/Resources.md): List of relevant resources

[Presentation](https://github.com/adamosSol/SC-DNN/blob/master/docs/Presentation.pdf): Project 15 minutes presentation 
