# DIRAC
Finding spin glass ground states through deep reinforcement learning
## Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Reproduction instructions](#reproduction-instructions)

# Overview

Spin glasses are disordered magnets with random interactions that are, generally, in conflict with each other. Finding the ground states of spin glasses is not only essential for understanding the nature of disordered magnetic and many other physical systems, but also useful to solve a broad array of hard combinatorial optimization problems across multiple disciplines. Despite decades-long efforts, an algorithm with both high accuracy and high efficiency is still lacking. Here we introduce DIRAC – a deep reinforcement learning framework, which can be trained purely on small-scale spin glass instances and then applied to arbitrarily large ones. DIRAC displays better scalability than other methods and can
be leveraged to enhance any thermal annealing method. Extensive calculations on 2D, 3D and 4D Edwards-Anderson spin glass instances demonstrate the superior performance of DIRAC over existing methods. The presented framework will help us better understand the nature of the low-temperature spin-glass phase, which is a fundamental challenge
in statistical physics. Moreover, the gauge transformation technique (which originates from physics) adopted in DIRAC builds a deep connection between physics and artificial intelligence, and opens up a promising avenue for reinforcement learning models to explore in the infinite configuration space, which would be extremely helpful to solve many other combinatorial hard problems.


# System Requirements

## Software dependencies and operating systems

### Software dependencies

Users should install the following packages first, which will install in about 5 minutes on a machine with the recommended specs. The versions of software are, specifically:
```
Cython==0.29.21
networkx==2.2
numpy==1.19.2 
pandas==0.25.2 
scipy==1.5.2
tensorflow-gpu==1.14.0 
tqdm==4.53.0
```

### Operating systems
The package development version is tested on *Linux and Windows 10* operating systems. The developmental version of the package has been tested on the following systems:

Linux: Ubuntu 18.04  
Windows: 10

The pip package should be compatible with Windows, and Linux operating systems.

Before setting up the DIRAC users should have `gcc` version 7.4.0 or higher.

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
### 1. Train the model, (e.g.)
```
CUDA_VISIBLE_DEVICES=gpu_id python train.py --lattice_dim 2 --lattice_num_min 10 --lattice_num_max 11 --model_name Lattice_2D
```
Modify the hyper-parameters in `SGRL.py` to tune the model, and make files after the the modification.

### 2. Test the model,

#### 2.1 Test using DIRAC^1 strategy
(unannotate DIRAC_1() in the main function)
```
CUDA_VISIBLE_DEVICES=-1 python test.py --lattice_dim 2 (e.g.) (do not use GPU for test)
```
Using the well-trained model (stored in `./models`), you can obtain the results reported in the paper.

#### 2.2 Test using DIRAC^m strategy
(unannotate DIRAC_m() in the main function)
```
CUDA_VISIBLE_DEVICES=-1 python test.py --lattice_dim 2
```

#### 2.3 Test using DIRAC_PT_alpha strategy
(unannotate DIRAC_PT_alpha() in the main function)
```
CUDA_VISIBLE_DEVICES=-1 python test.py --lattice_dim 2 --test_scale 15 --numInits 1000 --gid 0
```

#### 2.4 Test using DIRAC_PT_beta strategy
(unannotate DIRAC_PT_beta() in the main function)
```
CUDA_VISIBLE_DEVICES=-1 python test.py --lattice_dim 2 --test_scale 15 --numInits 1000 --gid 0
```

## Expected output
The experimental results are saved in the `results` folder, each strategy has a separate file to save the results.


# Reference

Please cite our work if you find our code/paper is useful to your work. 

@article{fan2021finding,
  title={Finding spin glass ground states through deep reinforcement learning},
  author={Fan, Changjun and Shen, Mutian and Nussinov, Zohar and Liu, Zhong and Sun, Yizhou and Liu, Yang-Yu},
  journal={arXiv preprint arXiv:2109.14411},
  year={2021}
}


