#!/bin/python

import numpy as np
import sys
import os
import bilby
import inspect
from enterprise_warp import enterprise_warp
from enterprise_warp import bilby_warp
from enterprise_warp.enterprise_warp import get_noise_dict
from enterprise_extensions import hypermodel

import ppta_dr2_models

opts = enterprise_warp.parse_commandline()

custom = ppta_dr2_models.PPTADR2Models

params = enterprise_warp.Params(opts.prfile,opts=opts,custom_models_obj=custom)
pta = enterprise_warp.init_pta(params)

if params.sampler == 'ptmcmcsampler':
    super_model = hypermodel.HyperModel(pta)
    print('Super model parameters: ', super_model.params)
    print('Output directory: ', params.output_dir)
    sampler = super_model.setup_sampler(resume=True, outdir=params.output_dir)
    N = params.nsamp
    x0 = super_model.initial_sample()
    noisedict = get_noise_dict(psrlist=[pp.name for pp in params.psrs],
                               noisefiles=params.noisefiles)
    x0 = super_model.informed_sample(noisedict)

    # Remove extra kwargs that Bilby took from PTSampler module, not ".sample"
    ptmcmc_sample_kwargs = inspect.getargspec(sampler.sample).args
    upd_sample_kwargs = {key: val for key, val in params.sampler_kwargs.items()
                                  if key in ptmcmc_sample_kwargs}
    del upd_sample_kwargs['Niter']
    del upd_sample_kwargs['p0']
    if opts.mpi_regime != 1:
      sampler.sample(x0, N, **upd_sample_kwargs)
    else:
      print('Preparations for the MPI run are complete - now set \
             opts.mpi_regime to 2 and enjoy the speed!')
else:
    priors = bilby_warp.get_bilby_prior_dict(pta[0])
    parameters = dict.fromkeys(priors.keys())
    likelihood = bilby_warp.PTABilbyLikelihood(pta[0],parameters)
    label = os.path.basename(os.path.normpath(params.out))
    if opts.mpi_regime != 1:
      bilby.run_sampler(likelihood=likelihood, priors=priors,
                        outdir=params.output_dir, label=params.label,
                        sampler=params.sampler, **params.sampler_kwargs)
    else:
      print('Preparations for the MPI run are complete - now set \
             opts.mpi_regime to 2 and enjoy the speed!')

with open(params.output_dir + "completed.txt", "a") as myfile:
  myfile.write("completed\n")
print('Finished: ',opts.num)
