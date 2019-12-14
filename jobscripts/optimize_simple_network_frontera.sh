#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export JOB_NAME=optimize_simple_network_"$1"_"$DATE"
export CONFIG_FILE=optimize_simple_network_"$1"_"$2"_config.yaml
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /scratch1/06441/aaronmil/src/optimize_cells/logs/$JOB_NAME.%j.o
#SBATCH -e /scratch1/06441/aaronmil/src/optimize_cells/logs/$JOB_NAME.%j.e
#SBATCH -p normal
#SBATCH -N 400
#SBATCH -n 22400
#SBATCH -t 6:00:00
#SBATCH --mail-user=aaronmil@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd $SCRATCH/src/optimize_cells

ibrun -n 22400 python3 -m nested.optimize --config-file-path=config/$CONFIG_FILE \
    --output-dir=data --pop_size=200 --max_iter=50 --path_length=3 --disp --procs_per_worker=112 --export
EOT
