# Search for the nanohertz stochastic gravitational-wave background with PPTA DR2

[arXiv:2107.12112](https://arxiv.org/abs/2107.12112) | [ApJL 917 L19](https://doi.org/10.3847/2041-8213/ac17f4)

This repository contains the code to reproduce results of the search for a stochastic gravitational-wave background with the second data release (DR2) of the Parkes Pulsar Timing Array (PPTA).
 - `run_analysis.py` is the main script to run parameter estimation and model selection.
 - `params/` contains confguration files for parameter estimation and/or model selection, priors, output directories, etc. 
 - `slurm/` contains slurm submission scripts, with examples of how to launch "run\_analysis.py" from the command line.

## Dependencies

The code is based on [enterprise_warp](https://github.com/bvgoncharov/enterprise_warp/). It was used with `Python 3.6`, `bilby-1.1.1`. It also requires a custom fork of `enterprise_extensions`, with some minor modifications. The **most important** differences are the hard-coded custom jump proposal uniform prior boundaries for the amplitude of gravitational-wave signal components in `enterprise_extensions`. Please make sure they are the same the actual prior boundaries for the analysis, as defined in `enterprise_warp/enterprise_models.py` or a parameter file where one can modify them. Otherwise, your will observe the biased posterior distribution.

## Posterior samples for the common-spectrum process

You are welcome to use posterior samples for the common-spectrum process, for `log10_A_gw` and `gamma_gw`. This repository contains samples for Figure 1, left, from the publication. For the solid blue contour. The samples are split between the two `.txt` files in `/publication_figures/`.

![The common-spectrum process and spatial correlations](https://github.com/bvgoncharov/correlated_noise_pta_2020/blob/master/publication_figures/illustration.jpg "The common-spectrum process and spatial correlations (arXiv: 2107.12112)")

## Citation

> @misc{goncharov2021evidence,\
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;title={On the evidence for a common-spectrum process in the search for the nanohertz gravitational-wave background with the Parkes Pulsar Timing Array},\
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;author={Boris Goncharov and R. M. Shannon and D. J. Reardon and G. Hobbs and A. Zic and M. Bailes and M. Curylo and S. Dai and M. Kerr and M. E. Lower and R. N. Manchester and R. Mandow and H. Middleton and M. T. Miles and A. Parthasarathy and E. Thrane and N. Thyagarajan and X. Xue and X. J. Zhu and A. D. Cameron and Y. Feng and R. Luo and C. J. Russell and J. Sarkissian and R. Spiewak and S. Wang and J. B. Wang and L. Zhang and S. Zhang},\
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;journal = {\apjl},\
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;year={2021},\
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;volume = {917},\
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;number = {2},\
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pages = {8},\
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;doi = {10.3847/2041-8213/ac17f4},\
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;eprint={2107.12112},\
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;archivePrefix={arXiv},\
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;primaryClass={astro-ph.HE}\
> }
