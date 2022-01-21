# install requirements
pip install -r requirements.txt

# compile
python setup.py build_ext -i

# train
(Add “CUDA_VISIBLE_DEVICES=gpu_id”beforehand for GPU training, otherwise use“CUDA_VISIBLE_DEVICES=-1”for CPU training)
e.g., python train.py --lattice_dim 2 --lattice_num_min 10 --lattice_num_max 11 --model_name Lattice_2D


# test
(Add “CUDA_VISIBLE_DEVICES=gpu_id”beforehand for GPU test, otherwise use“CUDA_VISIBLE_DEVICES=-1”for CPU test)

## test DIRAC^1
(unannotate DIRAC_1() in the main function)
e.g., python test.py --lattice_dim 2

## test DIRAC^m
(unannotate DIRAC_m() in the main function)
e.g., python test.py  --lattice_dim 2

## test DIRAC_PT_alpha
(unannotate DIRAC_PT_alpha() in the main function)
e.g., python test.py --lattice_dim 2 --test_scale 15 --numInits 1000 --gid 0

## test DIRAC_PT_beta
(unannotate DIRAC_PT_beta() in the main function)
e.g., python test.py --lattice_dim 2 --test_scale 15 --numInits 1000 --gid 0
