#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export JOB_NAME=optimize_DG_GC_excitability_gid_0_"$DATE"
export STORAGE_FILE_PATH=$1
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /scratch1/06441/aaronmil/src/optimize_cells/logs/$JOB_NAME.%j.o
#SBATCH -e /scratch1/06441/aaronmil/src/optimize_cells/logs/$JOB_NAME.%j.e
#SBATCH -p normal
#SBATCH -N 22
#SBATCH -n 1232
#SBATCH -t 12:00:00
#SBATCH --mail-user=aaronmil@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd $SCRATCH/src/optimize_cells

ibrun -n 1232 python3 -m nested.optimize --config-file-path=config/optimize_DG_GC_excitability_config.yaml \
    --output-dir=data --pop_size=200 --max_iter=50 --path_length=3 --disp --label=gid_0 --verbose=1 --hot-start \
    --history-file-path=$STORAGE_FILE_PATH
EOT
