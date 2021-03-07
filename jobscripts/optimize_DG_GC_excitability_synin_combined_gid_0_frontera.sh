#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export JOB_NAME=optimize_DG_GC_synaptic_integration_gid_0_"$DATE"
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /scratch1/06441/aaronmil/logs/optimize_cells/$JOB_NAME.%j.o
#SBATCH -e /scratch1/06441/aaronmil/logs/optimize_cells/$JOB_NAME.%j.e
#SBATCH -p normal
#SBATCH -N 36
#SBATCH -n 2016
#SBATCH -t 12:00:00
#SBATCH --mail-user=neurosutras@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd $WORK/optimize_cells

ibrun -n 2016 python3 -m nested.optimize \
    --config-file-path=config/optimize_DG_GC_excitability_synint_combined_gid_0_config.yaml \
    --output-dir=$SCRATCH/data/optimize_cells --pop_size=200 --max_iter=50 --path_length=3 --disp \
    --label=gid_0 --verbose=1 --dataset_prefix=$SCRATCH/data/dentate
EOT
