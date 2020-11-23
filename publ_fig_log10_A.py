import numpy as np
from matplotlib import pyplot as plt

from enterprise_warp.results import parse_commandline
from enterprise_warp.results import EnterpriseWarpResult
from make_factorized_posterior import FactorizedPosteriorResult

# What results to grab
psrs_set = '/home/bgonchar/correlated_noise_pta_2020/params/pulsar_set_x_1.dat'
output_directory = '/home/bgonchar/correlated_noise_pta_2020/publication_figures/'
result = [
'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_cpl_gwb_noauto_fixslope_20_nf_set_x_1_ephem_0_20201109.dat',
'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_cpl_gwb_noauto_fixslope_20_nf_set_x_1_ephem_0_20201109.dat',
#'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_ms_common_pl_set_x_1_ephem_0_20201103.dat'
]
par = [
['gw_log10_A_hd'],
['gw_log10_A'],
#['gw_log10_A'],
]
nmodel = [
0,
0,
#1
]
labels = [
"HD sampled with CPL",
"CPL sampled with HD",
#"CPL"
]
colors = [
"purple",
"#F39C12",
#"#E39C12"
]

hatch = [
None,
None,
#"/"
]

opts = parse_commandline()
opts.__dict__['logbf'] = True

#plt.style.use('seaborn-white')
fig, axes = plt.subplots()
for rr, pp, nm, ll, cc, hh in zip(result, par, nmodel, labels, colors, hatch):
  opts.__dict__['result'] = rr
  opts.__dict__['par'] = pp

  result_obj = EnterpriseWarpResult(opts)
  result_obj.main_pipeline()
  model_mask = np.round(result_obj.chain_burn[:,result_obj.ind_model]) == nm
  values = result_obj.chain_burn[model_mask,:]
  values = values[:,result_obj.par_mask]
  plt.hist(values, bins=20, density=True, histtype='stepfilled', alpha=0.5, facecolor=cc, hatch=hh, edgecolor=cc, label = ll)
plt.xlabel('$\log_{10}A$')
plt.ylabel('Posterior probability')
#plt.xlim([-17,-12])
#plt.ylim([1e-6, 8])
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig(output_directory + 'log10_A_hd.pdf')
plt.close()
