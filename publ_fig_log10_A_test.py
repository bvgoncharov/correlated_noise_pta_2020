import numpy as np
from matplotlib import pyplot as plt

from enterprise_warp.results import parse_commandline
from enterprise_warp.results import EnterpriseWarpResult
from make_factorized_posterior import FactorizedPosteriorResult

# What results to grab
psrs_set = '/home/bgonchar/correlated_noise_pta_2020/params/pulsar_set_3_1.dat'
output_directory = '/home/bgonchar/correlated_noise_pta_2020/publication_figures/'
result = [
'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_wnfix_pe_common_pl_factorized_20200916.dat',
#'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_ms_common_pl_vargam_set_x_1_ephem_0_20200928.dat',
'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_common_pl_fixgam_set_3_1_ephem_0_20201104.dat'
]
par = [
['gw'],
#['gw_log10_A'],
['gw_log10_A'],
]
nmodel = [
None,
#1,
0
]
labels = [
"CPL factorized",
#"$\mathrm(CPL, 30 bins, }\gamma\mathrm{-marg.}$",
"CPL"
]

opts = parse_commandline()
opts.__dict__['logbf'] = True

fig, axes = plt.subplots()
for rr, pp, nm, ll in zip(result, par, nmodel, labels):
  opts.__dict__['result'] = rr
  opts.__dict__['par'] = pp
  if 'factorized' in rr:
    result_obj = FactorizedPosteriorResult(opts, psrs_set=psrs_set)
    result_obj.main_pipeline([-20., -6.], plot_results = False)
    fobj = result_obj.make_figure(axes)
  else:
    result_obj = EnterpriseWarpResult(opts)
    result_obj.main_pipeline()
    model_mask = np.round(result_obj.chain_burn[:,result_obj.ind_model]) == nm
    values = result_obj.chain_burn[model_mask,:]
    values = values[:,result_obj.par_mask]
    plt.hist(values, bins=40, density=True, label = ll)
plt.xlabel('$\log_{10}A$')
plt.ylabel('Posterior probability')
#plt.xlim([-18,-14])
plt.legend()
plt.tight_layout()
plt.savefig(output_directory + 'log10_A.pdf')
plt.close()
