#!/bin/bash -l

#SBATCH -J optimize_simple_network_I_20191107
#SBATCH -o /global/cscratch1/sd/aaronmil/optimize_cells/optimize_simple_network_I_20191107.%j.o
#SBATCH -e /global/cscratch1/sd/aaronmil/optimize_cells/optimize_simple_network_I_20191107.%j.e
#SBATCH -q premium
#SBATCH -N 400 -n 12800
#SBATCH -L SCRATCH
#SBATCH -C haswell
#SBATCH -t 04:00:00
#SBATCH --mail-user=aaronmil@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd $HOME/optimize_cells

srun -n 12800 -c 2 python -m nested.optimize --config-file-path=config/optimize_simple_network_I_uniform_c_structured_w_gaussian_inp_config.yaml --output-dir=$SCRATCH/optimize_cells --pop_size=200 --max_iter=50 --path_length=3 --disp --procs_per_worker=64 --export
