# Search for the nanohertz stochastic gravitational-wave background with PPTA DR2

[arXiv:2107.12112](https://arxiv.org/abs/2107.12112)

This repository contains the code to reproduce results of the search for a stochastic gravitational-wave background with the second data release (DR2) of the Parkes Pulsar Timing Array (PPTA).
 - `run_analysis.py` is the main script to run parameter estimation and model selection.
 - `params/` contains confguration files for parameter estimation and/or model selection, priors, output directories, etc. 
 - `slurm/` contains slurm submission scripts, with examples of how to launch "run\_analysis.py" from the command line.

## Dependencies

The code is based on [enterprise_warp](https://github.com/bvgoncharov/enterprise_warp/). It was used with `Python 3.6`, `bilby-1.1.1`. It also requires a custom fork of `enterprise_extensions`, with some minor modifications.

## Citation

> @misc{goncharov2021evidence,\
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;title={On the evidence for a common-spectrum process in the search for the nanohertz gravitational wave background with the Parkes Pulsar Timing Array},\
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;author={Boris Goncharov and R. M. Shannon and D. J. Reardon and G. Hobbs and A. Zic and M. Bailes and M. Curylo and S. Dai and M. Kerr and M. E. Lower and R. N. Manchester and R. Mandow and H. Middleton and M. T. Miles and A. Parthasarathy and E. Thrane and N. Thyagarajan and X. Xue and X. J. Zhu and A. D. Cameron and Y. Feng and R. Luo and C. J. Russell and J. Sarkissian and R. Spiewak and S. Wang and J. B. Wang and L. Zhang and S. Zhang},\
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;year={2021},\
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;eprint={2107.12112},\
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;archivePrefix={arXiv},\
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;primaryClass={astro-ph.HE}\
> }

The paper is accepted in the Astrophysical Journal Letters, the citation to be updated.
