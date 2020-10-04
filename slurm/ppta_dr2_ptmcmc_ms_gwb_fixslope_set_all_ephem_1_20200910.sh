#!/bin/bash
#SBATCH --job-name=ppta_ptmcmc_ms_gwb_fixslope_set_all_ephem_1
#SBATCH --output=/fred/oz002/bgoncharov/correlated_noise_logs/ppta_ptmcmc_ms_gwb_fixslope_set_all_ephem_1_%A_%a.out
#SBATCH --ntasks=64
#SBATCH --time=0-12
#SBATCH --mem-per-cpu=3G
#SBATCH --tmp=4G
#SBATCH --array=0

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/correlated_noise_pta_2020/run_analysis_informed_x0.py --prfile "/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_ms_gwb_fixslope_set_all_ephem_1_20200916.dat" --num $SLURM_ARRAY_TASK_ID
