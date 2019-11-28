#!/bin/bash -l

#SBATCH -J optimize_DG_MC_excitability_cell1000000_20191125
#SBATCH -o /scratch1/04119/pmoolcha/HDM/optimize_cells/logs/optimize_DG_MC_excitability_cell1000000_20191125.%j.o
#SBATCH -e /scratch1/04119/pmoolcha/HDM/optimize_cells/logs/optimize_DG_MC_excitability_cell1000000_20191125.%j.e
#SBATCH -p development 
#SBATCH -N 22 -n 1232 
#SBATCH -t 1:00:00
#SBATCH --mail-user=pmoolcha@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd /scratch1/04119/pmoolcha/HDM/optimize_cells 

#srun -N 22 -n 1200 -c 2 python3 -m nested.optimize --config-file-path=config/optimize_DG_GC_excitability_config.yaml --pop_size=200 --max_iter=50 --path_length=3 --disp --output-dir=$SCRATC#H/optimize_cells --label=cell0 --export

ibrun gdb --batch -x /scratch1/04119/pmoolcha/HDM/optimize_cells/gdb_script --args python3 -m nested.optimize --config-file-path='config/optimize_DG_MC_excitability_config.yaml' --pop_size=200 --max_iter=2 --path_length=1 --disp --output-dir=data --label=cell1000000 --export
 
