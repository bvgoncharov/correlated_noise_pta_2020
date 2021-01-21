#!/bin/bash
#SBATCH --job-name=ppta_ptmcmc_pe
#SBATCH --output=/fred/oz002/bgoncharov/correlated_noise_logs/ppta_ptmcmc_pe_%A_%a.out
#SBATCH --ntasks=2
#SBATCH --time=1-23
#SBATCH --mem-per-cpu=2G
#SBATCH --tmp=2G
#SBATCH --array=8,11,23

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/correlated_noise_pta_2020/run_analysis.py --prfile "/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_20201215.dat" --num $SLURM_ARRAY_TASK_ID
