#!/bin/bash -l

#SBATCH -J optimize_DG_HICAP_Izhi_leak_spiking_cell1000000
#SBATCH -o /scratch1/04119/pmoolcha/HDM/optimize_cells/logs/optimize_DG_HICAP_Izhi_leak_spike_cell1043250_20200724.%j.o
#SBATCH -e /scratch1/04119/pmoolcha/HDM/optimize_cells/logs/optimize_DG_HICAP_Izhi_leak_spike_cell1043250_20200724.%j.e
#SBATCH -p normal 
#SBATCH -N 22 -n 1232 
#SBATCH -t 0:30:00
#SBATCH -A BIR20001
#SBATCH --mail-user=pmoolcha@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

source ~/.modules
cd /scratch1/04119/pmoolcha/HDM/optimize_cells 

#ibrun gdb --batch -x /scratch1/04119/pmoolcha/HDM/optimize_cells/gdb_script --args python3 -m nested.optimize --config-file-path='config/optimize_DG_MC_excitability_config.yaml' --pop_size=200 --max_iter=2 --path_length=1 --disp --output-dir=data --label=cell1000000 --export
 

ibrun python3 -m nested.optimize --config-file-path='config/optimize_DG_HICAP_Izhi_leak_spiking_config.yaml' --pop_size=200 --max_iter=50 --path_length=3 --disp --output-dir=data --label=cell1043250
