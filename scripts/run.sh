#!/bin/bash

# 32^3
mpirun -np 4 ./project_fv.py -r 32 --order 2 -f ../hittools/data/hit_ut_wavespace_256.npz

# 64^3
mpirun -np 8 ./project_fv.py -r 64 --order 2 -f ../hittools/data/hit_ut_wavespace_256.npz

# 128^3
mpirun -np 16 ./project_fv.py -r 128 --order 2 -f ../hittools/data/hit_ut_wavespace_256.npz

# 256^3
mpirun -np 16 ./project_fv.py -r 256 --order 2 -f ../hittools/data/hit_ut_wavespace_256.npz

# 512^3
mpirun -np 32 ./project_fv.py -r 512 --order 2 -f ../hittools/data/hit_ut_wavespace_256.npz

# 1024^3
mpirun -np 32 ./project_fv.py -r 1024 --order 2 -f ../hittools/data/hit_ut_wavespace_256.npz
