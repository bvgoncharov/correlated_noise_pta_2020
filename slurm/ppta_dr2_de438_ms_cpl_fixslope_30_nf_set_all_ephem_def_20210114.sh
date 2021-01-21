#!/bin/bash
#SBATCH --job-name=de438_ms_cpl_fixsl_30nf_setall_ephd
#SBATCH --output=/fred/oz002/bgoncharov/correlated_noise_logs/de438_ptmcmc_ms_cpl_fixslope_30_nf_set_all_ephem_def_%A_%a.out
#SBATCH --ntasks=4
#SBATCH --time=1-23
#SBATCH --mem-per-cpu=7G
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
srun python /home/bgonchar/correlated_noise_pta_2020/run_analysis.py --prfile "/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_de438_ms_common_pl_30_nf_set_all_ephem_def_20210114.dat" --num $SLURM_ARRAY_TASK_ID
