#!/bin/bash
#SBATCH --job-name=de430_ms_set_all_ephem_jup_el_3
#SBATCH --output=/fred/oz002/bgoncharov/correlated_noise_logs/de430_ptmcmc_ms_set_all_ephem_jup_el_3_%A_%a.out
#SBATCH --ntasks=4
#SBATCH --time=1-11
#SBATCH --mem-per-cpu=5G
#SBATCH --tmp=10G
#SBATCH --array=0

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/correlated_noise_pta_2020/run_analysis.py --prfile "/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_de430_ms_set_all_jup_el_3_20210114.dat" --num $SLURM_ARRAY_TASK_ID
