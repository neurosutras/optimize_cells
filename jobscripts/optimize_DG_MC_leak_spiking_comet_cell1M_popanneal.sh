#!/bin/bash -l

#SBATCH -J optimize_DG_MC_leak_spiking_shortdend_cell1000000_nearzerobounds_popanneal
#SBATCH -o ./logs/optimize_DG_MC_leak_spiking_cell1000000_20200402.%j.o
#SBATCH -e ./logs/optimize_DG_MC_leak_spiking_cell1000000_20200402.%j.e
#SBATCH -p compute
#SBATCH -N 50 
#SBATCH --ntasks-per-node=24
#SBATCH -t 15:00:00
#SBATCH --mail-user=pmoolcha@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

source ~/.modules
cd /home/pmoolcha/Work/optimize_cells

#srun -N 22 -n 1200 -c 2 python3 -m nested.optimize --config-file-path=config/optimize_DG_GC_excitability_config.yaml --pop_size=200 --max_iter=50 --path_length=3 --disp --output-dir=$SCRATC#H/optimize_cells --label=cell0 --export

#ibrun gdb --batch -x /scratch1/04119/pmoolcha/HDM/optimize_cells/gdb_script --args python3 -m nested.optimize --config-file-path='config/optimize_DG_MC_excitability_config.yaml' --pop_size=200 --max_iter=2 --path_length=1 --disp --output-dir=data --label=cell1000000 --export
 

#ibrun python3 -m nested.optimize --config-file-path='config/optimize_DG_MC_leak_spiking_config.yaml' --pop_size=200 --max_iter=50 --path_length=3 --disp --output-dir=data --label=cell1000000 --export

#ibrun python3 -m nested.optimize --config-file-path='config/optimize_DG_MC_leak_spiking_shortdend_config.yaml' --pop_size=200 --max_iter=50 --path_length=3 --disp --output-dir=data --label=cell1000000 --export

#mpirun -ib python3 -m nested.optimize --config-file-path='config/optimize_DG_MC_leak_spiking_shortdend_nearzero_popanneal_config.yaml' --pop_size=200 --max_iter=50 --path_length=3 --disp --output-dir=data --label=cell1000000 --export

mpirun -ib python3 -m nested.optimize --config-file-path='config/optimize_DG_MC_leak_spiking_shortdend_config.yaml' --pop_size=200 --max_iter=50 --path_length=3 --disp --output-dir=data --label=cell1000000 --export

#mpirun -ib python3 -m nested.optimize --config-file-path='config/optimize_DG_MC_leak_spiking_shortdend_config.yaml' --pop_size=200 --max_iter=50 --path_length=3 --disp --output-dir=data --label=cell1000000 --export --hot-start --history-file-path='data/20200414_0827_DG_MC_leak_spiking_cell1000000_PopulationAnnealing_optimization_history.hdf5'

#ibrun python3 -m nested.optimize --config-file-path='config/optimize_DG_MC_leak_spiking_shortdend_M_curr_soma_config.yaml' --pop_size=200 --max_iter=50 --path_length=3 --disp --output-dir=data --label=cell1000000 --export --hot-start --history-file-path='data/20200129_1429_DG_MC_leak_spiking_cell1000000_PopulationAnnealing_optimization_history.hdf5'
