# Search for the nanohertz stochastic gravitational-wave background with PPTA DR2

This repository contains the code to reproduce results of the search for a stochastic gravitational-wave background with the second data release (DR2) of the Parkes Pulsar Timing Array (PPTA).
 - `run_analysis.py` is the main script to run parameter estimation and model selection.
 - `params/` contains confguration files for parameter estimation and/or model selection, priors, output directories, etc. 
 - `slurm/` contains slurm submission scripts, with examples of how to launch "run\_analysis.py" from the command line.

## Dependencies

The code is based on [enterprise_warp](https://github.com/bvgoncharov/enterprise_warp/). It was used with `Python 3.6`, `bilby-1.1.1`. It also requires a custom fork of `enterprise_extensions`, with some minor modifications.

## Citation

> Goncharov, Shannon, Reardon, Hobbs, Zic, et al. (PPTA). On the evidence for a common-spectrum process in the search for the nanohertz gravitational wave background with the Parkes Pulsar Timing Array.

The paper is submitted to Astrophysical Journal Letters, citation example will be added upon acceptance.
