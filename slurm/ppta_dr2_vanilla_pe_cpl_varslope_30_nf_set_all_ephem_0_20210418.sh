#!/bin/bash
#SBATCH --job-name=ppta_van_pe_cpl_varsl_30nf_setall_eph0
#SBATCH --output=/fred/oz002/bgoncharov/correlated_noise_logs/ppta_vanilla_pe_cpl_varslope_30_nf_set_all_ephem_0_%A_%a.out
#SBATCH --ntasks=4
#SBATCH --time=1-11
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
srun python /home/bgonchar/correlated_noise_pta_2020/run_analysis.py --prfile "/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_vanilla_pe_common_pl_vargam_30_nf_set_all_ephem_0_20210418.dat" --num $SLURM_ARRAY_TASK_ID
