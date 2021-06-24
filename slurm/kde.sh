#!/bin/bash
#SBATCH --job-name=kde_ppta
#SBATCH --output=/fred/oz002/bgoncharov/correlated_noise_logs/ppta_kde_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --time=0-12
#SBATCH --mem-per-cpu=8G
#SBATCH --tmp=8G
#SBATCH --array=0

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/correlated_noise_pta_2020/publ_fig_dropout_varg.py
