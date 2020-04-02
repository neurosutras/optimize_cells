#!/bin/bash -l

#SBATCH -J optimize_DG_MC_leak_spiking_shortdend_cell1000000_nearzerobounds_sobol
#SBATCH -o /scratch1/04119/pmoolcha/HDM/optimize_cells/logs/optimize_DG_MC_excitability_cell1000000_20200316.%j.o
#SBATCH -e /scratch1/04119/pmoolcha/HDM/optimize_cells/logs/optimize_DG_MC_excitability_cell1000000_20200316.%j.e
#SBATCH -p normal 
#SBATCH -N 22 -n 1232 
#SBATCH -t 15:00:00
#SBATCH -A BIR20001
#SBATCH --mail-user=pmoolcha@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

source ~/.modules
cd /scratch1/04119/pmoolcha/HDM/optimize_cells 

#srun -N 22 -n 1200 -c 2 python3 -m nested.optimize --config-file-path=config/optimize_DG_GC_excitability_config.yaml --pop_size=200 --max_iter=50 --path_length=3 --disp --output-dir=$SCRATC#H/optimize_cells --label=cell0 --export

#ibrun gdb --batch -x /scratch1/04119/pmoolcha/HDM/optimize_cells/gdb_script --args python3 -m nested.optimize --config-file-path='config/optimize_DG_MC_excitability_config.yaml' --pop_size=200 --max_iter=2 --path_length=1 --disp --output-dir=data --label=cell1000000 --export
 

#ibrun python3 -m nested.optimize --config-file-path='config/optimize_DG_MC_leak_spiking_config.yaml' --pop_size=200 --max_iter=50 --path_length=3 --disp --output-dir=data --label=cell1000000 --export

#ibrun python3 -m nested.optimize --config-file-path='config/optimize_DG_MC_leak_spiking_shortdend_config.yaml' --pop_size=200 --max_iter=50 --path_length=3 --disp --output-dir=data --label=cell1000000 --export

#ibrun python3 -m nested.optimize --config-file-path='config/optimize_DG_MC_leak_spiking_shortdend_nearzero_sobol_config.yaml' --pop_size=200 --max_iter=50 --path_length=3 --disp --output-dir=data --label=cell1000000 --export

#ibrun python3 -m nested.optimize --config-file-path='config/optimize_DG_MC_leak_spiking_shortdend_nearzero_sobol_config.yaml' --output-dir=data --label=cell1000000 --export --num_models=30000 --param-gen=Sobol --hot-start --storage-file-path=data/20200325_1638_DG_MC_leak_spiking_Sobol_optimization_history.hdf5 --pregen_param_file_path=data/20200325_1638_Sobol_sequence.hdf5 --verbose=2

#ibrun python3 -m nested.optimize --config-file-path='config/optimize_DG_MC_leak_spiking_shortdend_M_curr_soma_config.yaml' --pop_size=200 --max_iter=50 --path_length=3 --disp --output-dir=data --label=cell1000000 --export --hot-start --storage-file-path='data/20200129_1429_DG_MC_leak_spiking_cell1000000_PopulationAnnealing_optimization_history.hdf5'

ibrun python3 -m nested.optimize --config-file-path='data/20200331_1820_20200327_1848_optimize_DG_MC_leak_spiking_shortdend_nearzero_sobol_config.yaml' --output-dir=data --label=cell1000000 --export --num_models=30000 --param-gen=Sobol --hot-start --storage-file-path=data/20200331_1820_DG_MC_leak_spiking_cell1000000_Sobol_optimization_history.hdf5 --pregen_param_file_path=data/20200331_1820_Sobol_sequence.hdf5 --verbose=0 --pop_size=100
