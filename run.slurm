#!/bin/bash
#SBATCH --job-name=detection_fractions     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=1G             # memory per cpu-core (4G is default)
#SBATCH --time=4:00:00          # total run time limit (HH:MM:SS)  HAD -r 0.0001
#SBATCH -p conroy,itc_cluster,shared,sapphire #partition to submit to
#SBATCH --mail-type=END
#SBATCH --output=slurm-ms-%j.out

module purge
module load python/3.12.5-fasrc01
module load gcc/14.2.0-fasrc01 gsl/2.8-fasrc01
mamba activate petar_env

python -u mock_obs.py

