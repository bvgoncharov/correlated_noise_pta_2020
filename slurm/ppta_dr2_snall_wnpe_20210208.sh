#!/bin/bash
#SBATCH --job-name=ppta_snall_wnpe
#SBATCH --output=/fred/oz002/bgoncharov/correlated_noise_logs/ppta_snall_wnpe_%A_%a.out
#SBATCH --ntasks=4
#SBATCH --time=1-21
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=8G
#SBATCH --array=15,8,9

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/correlated_noise_pta_2020/run_analysis.py --prfile "/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_snall_wnpe_20210208.dat" --num $SLURM_ARRAY_TASK_ID
