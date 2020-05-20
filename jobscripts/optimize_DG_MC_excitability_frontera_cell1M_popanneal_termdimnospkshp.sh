#!/bin/bash -l

#SBATCH -J optimize_DG_MC_excitability_shortdendnewtermdimnospkshp_cell1000000_earlier_bounds_popanneal
#SBATCH -o /scratch1/04119/pmoolcha/HDM/optimize_cells/logs/optimize_DG_MC_excitability_cell1000000_20200504.%j.o
#SBATCH -e /scratch1/04119/pmoolcha/HDM/optimize_cells/logs/optimize_DG_MC_excitability_cell1000000_20200504.%j.e
#SBATCH -p normal 
#SBATCH -N 50 -n 2800 
#SBATCH -t 2:00:00
#SBATCH -A BIR20001
#SBATCH --mail-user=pmoolcha@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

source ~/.modules
cd /scratch1/04119/pmoolcha/HDM/optimize_cells 

#srun -N 22 -n 1200 -c 2 python3 -m nested.optimize --config-file-path=config/optimize_DG_GC_excitability_config.yaml --pop_size=200 --max_iter=50 --path_length=3 --disp --output-dir=$SCRATC#H/optimize_cells --label=cell0 --export

#ibrun gdb --batch -x /scratch1/04119/pmoolcha/HDM/optimize_cells/gdb_script --args python3 -m nested.optimize --config-file-path='config/optimize_DG_MC_excitability_config.yaml' --pop_size=200 --max_iter=2 --path_length=1 --disp --output-dir=data --label=cell1000000 --export
 

#ibrun python3 -m nested.optimize --config-file-path='config/optimize_DG_MC_leak_spiking_config.yaml' --pop_size=200 --max_iter=50 --path_length=3 --disp --output-dir=data --label=cell1000000 --export

ibrun python3 -m nested.optimize --config-file-path='config/optimize_DG_MC_excitability_shortdendnewtermdimnospkshape_config.yaml' --pop_size=200 --max_iter=50 --path_length=3 --disp --output-dir=data --label=cell1000000 --hot-start --storage-file-path='data/20200512_0050_DG_MC_leak_spiking_cell1000000_PopulationAnnealing_optimization_history.hdf5'

#ibrun python3 -m nested.optimize --config-file-path='config/optimize_DG_MC_leak_spiking_shortdend_nearzero_popanneal_config.yaml' --pop_size=200 --max_iter=50 --path_length=3 --disp --output-dir=data --label=cell1000000 --export

#ibrun python3 -m nested.optimize --config-file-path='config/optimize_DG_MC_leak_spiking_shortdend_M_curr_soma_config.yaml' --pop_size=200 --max_iter=50 --path_length=3 --disp --output-dir=data --label=cell1000000 --export --hot-start --storage-file-path='data/20200129_1429_DG_MC_leak_spiking_cell1000000_PopulationAnnealing_optimization_history.hdf5'
