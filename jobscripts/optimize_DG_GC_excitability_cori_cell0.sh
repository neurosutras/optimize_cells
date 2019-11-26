#!/bin/bash -l

#SBATCH -J optimize_DG_GC_excitability_cell0_20181204
#SBATCH -o /global/cscratch1/sd/aaronmil/optimize_cells/optimize_DG_GC_excitability_cell0_20181204.%j.o
#SBATCH -e /global/cscratch1/sd/aaronmil/optimize_cells/optimize_DG_GC_excitability_cell0_20181204.%j.e
#SBATCH -q premium
#SBATCH -N 19 -n 608
#SBATCH -L SCRATCH
#SBATCH -C haswell
#SBATCH -t 12:00:00
#SBATCH --mail-user=aaronmil@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd $HOME/optimize_cells

srun -N 19 -n 608 -c 2 python -m nested.optimize --config-file-path=config/Cori_optimize_DG_GC_excitability_config.yaml --pop_size=200 --max_iter=50 --path_length=3 --disp --output-dir=$SCRATCH/optimize_cells --label=cell0 --export
