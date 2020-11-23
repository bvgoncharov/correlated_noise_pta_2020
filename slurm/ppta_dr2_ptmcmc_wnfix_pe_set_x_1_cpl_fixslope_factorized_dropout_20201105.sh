#!/bin/bash
#SBATCH --job-name=ppta_ptmcmc_x_1_pe_cpl_fixsl_factor_drop
#SBATCH --output=/fred/oz002/bgoncharov/correlated_noise_logs/ppta_ptmcmc_x_1_wnfix_pe_cpl_fixslope_factorized_dropout_%A_%a.out
#SBATCH --ntasks=4
#SBATCH --time=0-23
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=8G
#SBATCH --array=1-25

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/correlated_noise_pta_2020/run_analysis.py --prfile "/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_cpl_set_x_1_factorized_dropout_20201105.dat" --drop 1 --num $SLURM_ARRAY_TASK_ID
