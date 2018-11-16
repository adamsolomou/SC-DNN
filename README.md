# SC-DNN

## Overview
This repository provides source code and documentation for the implementation of Artificial Neural Networks (ANNs) using Stochastic Computing (SC), a novel computing paradigm that provides significantly lower hardware footprint compared to conventional binary computing. A software implementation of several computational elements is provided including scaled addition, multiplication, scaled inner product and a stochastic implementation of the hyperbolic function. Furthermore, a stochastic comparator is implemented and used to introduce a stochastic implementation of the rectified linear unit (ReLU). Saturation arithmetic elements are also introduced based on a stochastic linear gain function. A modified neuron architecture is proposed and implemented to model stochastic arithmetic during the training phase of a network.

## User Guide

Install dependencies listed in [requirements](https://github.com/adamosSol/SC-DNN/blob/master/requirements.txt). 

The script `scripts/load_mnist.py` loads the normalised MNIST dataset used for inference purposes. While being the `SC-DNN/` directory run 

    python scripts/load_mnist.py 

This saves the testing data features and labels in the `data/` directory. 

The directory `src/inference` contains source code for the implementation of neural network inference in stochastic computing. `src/inference/function_blocks.py` contains a (software) implementation of several SC processing elements employed in ANNs. `src/inference/scaling_ops.py` includes the implementation of the scaling operations that are used to determine signal scalings in the SC network graph during inference time. The directories `src/inference/network_I` and `src/inference/network_II` contain trained coefficients for the two test-case networks studied in this project. 

The code in `fp_inference.py` runs a forward propagation of the trained network in floating point arithmetic. The script `compute_scalings.py` computes the worst-case scaling parameters based on the trained coefficients whereas `signal_values.py` computes internal signal values and plots histograms in each layer, that can be used to determine scaling values for saturation arithmetic. The code in `sc_inference.py` runs a forward propagation of the trained network in stochastic arithmetic. Currently, network's parameters need to be manually imposed by the user. The numerical values for the parameters used in each experiment are listed in `sc_net_parameters.txt`. 

To run any of these scripts navigate to the corresponding repository and execute 

    python <name of script>
    
Note that if you try to execute a script that uses the function `scalar_next_int_power_of_2(x)` in `scaling_ops.py` while being in the TensorFlow environment, it will (most likely) raise an AttributeError: 'float' object has no attribute 'bit_length'. To deal with this, deactivate the TensorFlow environment and run the code again (Feel free to propose alternative solutions).  

The directory `src/training` contains source code for training SC compatible neural networks. Alternative matrix multiplication and addition procedures are implemented in `src/training/sc_train_creg.py` and `src/training/sc_train_l2reg.py`, to realize the proposed SC neuron architecture. The two Python files train a single hidden layer network on the MNIST dataset employing different regularization techniques. For these examples, an optimal regularization scale (mainly for L2 regularization) was found empirically. 

## Documentation
[Resources](https://github.com/adamosSol/SC-DNN/blob/master/docs/Resources.md): List of relevant resources

MEng Thesis available on request.  
