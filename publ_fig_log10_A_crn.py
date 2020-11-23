import numpy as np
from matplotlib import pyplot as plt

from enterprise_warp.results import parse_commandline
from enterprise_warp.results import EnterpriseWarpResult
from make_factorized_posterior import FactorizedPosteriorResult

# What results to grab
psrs_set = '/home/bgonchar/correlated_noise_pta_2020/params/pulsar_set_x_1.dat'
output_directory = '/home/bgonchar/correlated_noise_pta_2020/publication_figures/'
result = [
'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_common_pl_factorized_20200915.dat',
#'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_ms_common_pl_vargam_set_x_1_ephem_0_20200928.dat',
'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_ms_common_pl_set_x_1_ephem_0_20201103.dat'
]
par = [
['gw'],
#['gw_log10_A'],
['gw_log10_A'],
]
nmodel = [
None,
#1,
1
]
labels = [
"CPL factorized",
#"$\mathrm(CPL, 30 bins, }\gamma\mathrm{-marg.}$",
"CPL"
]

opts = parse_commandline()
opts.__dict__['logbf'] = True

#plt.style.use('seaborn-white')
ymin = 1e-12
ymax = 8
fig, axes = plt.subplots()
for rr, pp, nm, ll in zip(result, par, nmodel, labels):
  opts.__dict__['result'] = rr
  opts.__dict__['par'] = pp
  if 'factorized' in rr:
    result_obj = FactorizedPosteriorResult(opts, psrs_set=psrs_set)
    result_obj.main_pipeline([-20., -6.], plot_results = False)
    fobj = result_obj.make_figure(axes, label = ll)
  else:
    result_obj = EnterpriseWarpResult(opts)
    result_obj.main_pipeline()
    model_mask = np.round(result_obj.chain_burn[:,result_obj.ind_model]) == nm
    values = result_obj.chain_burn[model_mask,:]
    values = values[:,result_obj.par_mask]
    plt.hist(values, bins=20, density=True, histtype='stepfilled', alpha=0.5, facecolor='#F39C12', hatch='/', edgecolor='sienna', label = ll)
plt.vlines(np.log10(1.9e-15), ymin, ymax, linestyles="dotted", colors="black", label="NANOGrav 12.5-yr CPL")
plt.xlabel('$\log_{10}A$')
plt.ylabel('Posterior probability')
plt.xlim([-17,-12])
plt.ylim([ymin, ymax])
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig(output_directory + 'log10_A_crn.pdf')
plt.close()
