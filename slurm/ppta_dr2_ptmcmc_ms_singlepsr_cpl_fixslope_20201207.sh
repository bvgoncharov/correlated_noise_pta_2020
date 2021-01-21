#!/bin/bash
#SBATCH --job-name=ppta_ptmcmc_ms_singlepsr_cpl_fixsl
#SBATCH --output=/fred/oz002/bgoncharov/correlated_noise_logs/ppta_ptmcmc_ms_singlepsr_cpl_fixslope_%A_%a.out
#SBATCH --ntasks=4
#SBATCH --time=1-23
#SBATCH --mem-per-cpu=3G
#SBATCH --tmp=4G
#SBATCH --array=3-9,11,13-16,18,22-25

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/correlated_noise_pta_2020/run_analysis.py --prfile "/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_ms_singlepsr_common_pl_gfix_20201207.dat" --num $SLURM_ARRAY_TASK_ID
