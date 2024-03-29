#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export CONFIG_FILE_PATH=$1
export GID=$2
export LABEL=$3
export JOB_NAME=optimize_DG_GC_excitability_synint_combined_gid_"$GID"_"$LABEL"_"$DATE"
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /scratch1/06441/aaronmil/logs/optimize_cells/$JOB_NAME.%j.o
#SBATCH -e /scratch1/06441/aaronmil/logs/optimize_cells/$JOB_NAME.%j.e
#SBATCH -p development
#SBATCH -N 36
#SBATCH -n 2016
#SBATCH -t 1:00:00
#SBATCH --mail-user=neurosutras@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd $WORK2/optimize_cells

ibrun -n 2016 python3 -m nested.optimize \
    --config-file-path=$CONFIG_FILE_PATH \
    --output-dir=$SCRATCH/data/optimize_cells --pop_size=200 --max_iter=1 --path_length=1 --disp \
    --label=gid_"$GID"_"$LABEL" --verbose=1 --dataset_prefix=$SCRATCH/data/dentate --gid=$GID
EOT
