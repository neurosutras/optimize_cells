#!/bin/bash -l

#SBATCH -J optimize_DG_GC_synaptic_integration_burst_20190130
#SBATCH -o /global/cscratch1/sd/aaronmil/optimize_cells/optimize_DG_GC_synaptic_integration_burst_20190130.%j.o
#SBATCH -e /global/cscratch1/sd/aaronmil/optimize_cells/optimize_DG_GC_synaptic_integration_burst_20190130.%j.e
#SBATCH -q premium
#SBATCH -N 64 -n 2048
#SBATCH -L SCRATCH
#SBATCH -C haswell
#SBATCH -t 12:00:00
#SBATCH --mail-user=aaronmil@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#DW jobdw capacity=60GB access_mode=striped type=scratch
#DW stage_out source=$DW_JOB_STRIPED/optimize_cells destination=/global/cscratch1/sd/aaronmil/optimize_cells type=directory

set -x

mkdir $DW_JOB_STRIPED/optimize_cells

cd $HOME/optimize_cells

export HDF5_USE_FILE_LOCKING=FALSE

srun -N 64 -n 2048 -c 2 python -m nested.optimize --config-file-path=config/cori_optimize_DG_GC_synaptic_integration_config.yaml --output-dir=$DW_JOB_STRIPED/optimize_cells --pop_size=200 --max_iter=50 --path_length=3 --disp --label=cell0 --export
