#!/bin/bash
#SBATCH --job-name=ppta_30nf_all_pe_cplfixsl_factor_drop
#SBATCH --output=/fred/oz002/bgoncharov/correlated_noise_logs/ppta_ptmcmc_30nf_all_wnfix_pe_cpl_fixsl_factorz_dropout_%A_%a.out
#SBATCH --ntasks=4
#SBATCH --time=1-23
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=8G
#SBATCH --array=0-25

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/correlated_noise_pta_2020/run_analysis.py --prfile "/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_cpl_30_nf_set_all_factorized_dropout_20201219.dat" --drop 1 --num $SLURM_ARRAY_TASK_ID
