#!/bin/python

import pandas as pd
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
    # Filling in PTMCMCSampler jump covariance matrix
    if params.mcmc_covm is not None:
      ndim = len(super_model.param_names)
      identity_covm = np.diag(np.ones(ndim) * 1**2)
      identity_covm_df = pd.DataFrame(identity_covm, \
                                      index=super_model.param_names, \
                                      columns=super_model.param_names)
      covm = params.mcmc_covm.combine_first(identity_covm_df)
      identity_covm_df.update(covm)
      covm = np.array(identity_covm_df)
    else:
      covm = None
    sampler = super_model.setup_sampler(resume=True, outdir=params.output_dir, initial_cov=covm)
    N = params.nsamp
    x0 = super_model.initial_sample()
    try:
      noisedict = get_noise_dict(psrlist=[pp.name for pp in params.psrs],
                                 noisefiles=params.noisefiles)
      x0 = super_model.informed_sample(noisedict)
      # Start ORF inference with zero correlation:
      if 'corr_coeff_0' in super_model.param_names and \
         len(x0)==len(super_model.param_names):
        print('Starting sampling free ORF with zeros')
        orf_mask = [True if 'corr_coeff' in prn else False \
                    for prn in super_model.param_names]
        x0[orf_mask] = 0.
    except:
      print('Informed sample is not possible')

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
