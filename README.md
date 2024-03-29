# DIRAC (Deep reinforcement learning for spIn-glass gRound stAte Calculation)

This is a TensorFlow implementation of DIRAC, as described in our paper:

Fan, C., Shen, M., Nussinov, Z., Liu, Z., Sun, Y and Liu Y-Y. [Searching for spin glass ground states through deep reinforcement]. Nat. Commun. (2023). ![demo](https://github.com/FFrankyy/DIRAC/blob/main/Paper/Featured_Image_NC.png)

## Contents

- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Reproduction instructions](#reproduction-instructions)
- [Basebline methods implementation](#basebline-methods-implementation)

# Overview

Spin glasses are disordered magnets with random interactions that are, generally, in conflict with each other. Finding the ground states of spin glasses is not only essential for understanding the nature of disordered magnets and many other physical systems, but also useful to solve a broad array of hard combinatorial optimization problems across multiple disciplines. Despite decades-long efforts, an algorithm with both high accuracy and high efficiency is still lacking. Here we introduce DIRAC -- a deep reinforcement learning framework, which can be trained purely on small-scale spin glass instances and then applied to arbitrarily large ones. DIRAC displays better scalability than other methods and can be leveraged to enhance any thermal annealing method. Extensive calculations on 2D, 3D and 4D Edwards-Anderson spin glass instances demonstrate the superior performance of DIRAC over existing methods. The presented framework will help us better understand the nature of the low-temperature spin-glass phase, which is a fundamental challenge in statistical physics. Moreover, the gauge transformation technique adopted in DIRAC~builds a deep connection between physics and artificial intelligence. In particular, this opens up a promising avenue for reinforcement learning models to explore in the enormous configuration space, which would be extremely helpful to solve many other hard combinatorial optimization problems.

# Repo Contents

- [DIRAC](./DIRAC): source codes of DIRAC.
- [baselines](./baselines): implementation details of Simulated Annealing (SA) and Parallel Temperaturing (PT).


# System Requirements

## Software dependencies and operating systems

### Software dependencies

Users should install the following packages first, which will install in about 5 minutes on a machine with the recommended specs. The versions of software are, specifically:
```
cython==0.29.13 
networkx==2.3 
numpy==1.17.3 
pandas==0.25.2 
scipy==1.3.1 
tensorflow-gpu==1.14.0 
tqdm==4.36.1
```

### Operating systems
The package development version is tested on *Linux and Windows 10* operating systems. The developmental version of the package has been tested on the following systems:

Linux: Ubuntu 18.04  
Windows: 10

The pip package should be compatible with Windows, and Linux operating systems.

Before setting up the FINDER users should have `gcc` version 7.4.0 or higher.

## Hardware Requirements
The `DIRAC` model requires a standard computer with enough RAM and GPU to support the operations defined by a user. For minimal performance, this will be a computer with about 4 GB of RAM and 16GB of GPU. For optimal performance, we recommend a computer with the following specs:

RAM: 16+ GB  
CPU: 4+ cores, 3.3+ GHz/core
GPU: 16+ GB

The runtimes below are generated using a computer with the recommended specs (16 GB RAM, 4 cores@3.3 GHz) and internet of speed 25 Mbps.


# Installation Guide

## Instructions
1. First install all the above required packages, which are contained in the requirements.txt file
```
pip install -r requirements.txt
```
2. Make all the file
```
python setup.py build_ext -i
```

## Typical install time
It took about 5 mins to install all the required packages, and about 1 mins to make all the files.

# Reproduction instructions

## Instructions to run
1. Train the model, 
```
CUDA_VISIBLE_DEVICES=gpu_id python train.py --lattice_dim DIM --lattice_num_min NUM_MIN, --lattice_num_max NUM_MAX
```
Modify the hyper-parameters in `config.py` to train the model.

2. Test using $DIRAC^1$ strategy,
```
CUDA_VISIBLE_DEVICES=-1 python DIRAC1_test.py --lattice_dim DIM --test_scale SCALE (do not use GPU for test)
```
We provide the well-trained model (stored in `./models`), you can obtain the results reported in the paper. You can also specify the specific gpu_id to speed up the test with GPU.

3. Test using $DIRAC^m$ strategy,
```
CUDA_VISIBLE_DEVICES=-1 python DIRACm_test.py --lattice_dim DIM --test_scale SCALE (do not use GPU for test)
```
Using the well-trained model (stored in `./models`), you can obtain the results reported in the paper.

4. Test using DIRAC-SA strategy,
```
CUDA_VISIBLE_DEVICES=-1 python DIRAC-SA_test.py --lattice_dim DIM --test_scale SCALE --gid LATTICE_ID (do not use GPU for test)
```
Using the well-trained model (stored in `./models`), you can obtain the results reported in the paper.

5. Test using DIRAC-PT strategy,
```
CUDA_VISIBLE_DEVICES=-1 python DIRAC-PT_test.py --lattice_dim DIM --test_scale SCALE --args.numInits NUM_EPOCH --args.gid LATTICE_ID (do not use GPU for test)
```
Using the well-trained model (stored in `./models`), you can obtain the results reported in the paper.


# Basebline methods implementation
We compared with two competitive annealing-based algorithms, Simulated Annealing (SA) and Parallel Temperaturing (PT). We provide our implementations of these two algorithms, which are stored in `./baselines`.

The input always contains a matrix $J_{ij}$ that describes interaction between site $i$ and site $j$. There are also other arguments, such as random seed, or initial state as a vector.

The parameters of these two algorithms are from:

    SA: https://arxiv.org/abs/1412.2104
    
    PT: https://arxiv.org/abs/0806.1054

# Reproducing the results that reported in the paper

Here is the link to the dataset that was used in the paper, including the lattice files of different sides and different dimensions.

https://drive.google.com/file/d/1h2enhzjb1HOK6cbrrCo8yz9KYtZ6zjrH/view?usp=share_link

# Reference

Please cite our work if you find our code/paper is useful to your work. 
     
     @article{fan2021finding,
          title={Finding spin glass ground states through deep reinforcement learning},
          author={Fan, Changjun and Shen, Mutian and Nussinov, Zohar and Liu, Zhong and Sun, Yizhou and Liu, Yang-Yu},
          journal={arXiv preprint arXiv:2109.14411},
          year={2021}
        }
