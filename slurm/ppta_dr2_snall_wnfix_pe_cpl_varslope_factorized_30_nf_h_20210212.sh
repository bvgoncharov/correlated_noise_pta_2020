#!/bin/bash
#SBATCH --job-name=ppta_snall_wnfix_pe_cpl_varsl_factorz_30nf
#SBATCH --output=/fred/oz002/bgoncharov/correlated_noise_logs/ppta_snall_wnfix_pe_cpl_varslope_factorized_30nf_%A_%a.out
#SBATCH --ntasks=4
#SBATCH --time=1-23
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=5G
#SBATCH --array=0,13,20

pyv="$(python -c 'import sys; print(sys.version_info[0])')"
if [ "$pyv" == 2 ]
then
    echo "$pyv"
    module load numpy/1.16.3-python-2.7.14
fi

srun echo $TEMPO2
srun echo $TEMPO2_CLOCK_DIR
srun python /home/bgonchar/correlated_noise_pta_2020/run_analysis.py --prfile "/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_snall_wnfix_pe_common_pl_vargam_factorized_30_nf_20210212.dat" --num $SLURM_ARRAY_TASK_ID
