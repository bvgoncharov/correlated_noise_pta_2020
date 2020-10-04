#!/bin/bash
#SBATCH --job-name=ppta_ptmcmc_ms_cpl_fixslope_set_3_1_ephem_1
#SBATCH --output=/fred/oz002/bgoncharov/correlated_noise_logs/ppta_ptmcmc_ms_cpl_fixslope_set_3_1_ephem_1_%A_%a.out
#SBATCH --ntasks=4
#SBATCH --time=2-0
#SBATCH --mem-per-cpu=2G
#SBATCH --tmp=4G
#SBATCH --array=0

module load numpy/1.16.3-python-2.7.14

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/correlated_noise_pta_2020/run_analysis.py --prfile "/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_ms_common_pl_set_3_1_ephem_1_20200910.dat" --num $SLURM_ARRAY_TASK_ID
