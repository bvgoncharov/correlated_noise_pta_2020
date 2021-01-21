import numpy as np
from matplotlib import pyplot as plt

from enterprise_warp.results import parse_commandline
from enterprise_warp.results import EnterpriseWarpResult
from make_freespec_violin import ViolinFreespecResult

# What results to grab
output_directory = '/home/bgonchar/correlated_noise_pta_2020/publication_figures/'
result = [
'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_cpl_freesp_30_nf_set_3_1_ephem_0_20201015.dat',
'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_cpl_freesp_30_nf_set_3_1_ephem_c1_20201015.dat'
]
par = [
['gw_log10_rho'],
['gw_log10_rho']
]
nmodel = [
0,
0
]
labels = [
None,
None
]
colors = [
"purple",
"#F39C12",
#"#E39C12"
]
angles = [
None,
None
]


opts = parse_commandline()
opts.__dict__['logbf'] = True

#plt.style.use('seaborn-white')
fig, axes = plt.subplots()
for rr, pp, nm, ll, cc, aa in zip(result, par, nmodel, labels, colors, angles):
  opts.__dict__['result'] = rr
  opts.__dict__['par'] = pp

  result_obj = ViolinFreespecResult(opts)
  result_obj.main_pipeline(plot=False)
  fobj = result_obj.make_figure(axes, label = ll, color=cc)
  #model_mask = np.round(result_obj.chain_burn[:,result_obj.ind_model]) == nm
  #values = result_obj.chain_burn[model_mask,:]
  #values = values[:,result_obj.par_mask]
  #axes.violinplot(values, positions=aa, widths=np.sum(aa)/len(aa)/3, showextrema=False)
  #plt.hist(values, bins=20, density=True, histtype='stepfilled', alpha=0.5, facecolor=cc, hatch=hh, edgecolor=cc, label = ll)

axes.axvline(x=result_obj.inv_earth_orb_period)
axes.axvline(x=result_obj.inv_ven_orb_period)
axes.axvline(x=result_obj.inv_mar_orb_period)
axes.axvline(x=result_obj.inv_sat_orb_period)
axes.axvline(x=result_obj.inv_jup_orb_period)

plt.ylabel('$\\log_{10}(\\rho\\mathrm{[s]})$')
plt.xlabel('$\\mathrm{Frequency,~[Hz]}$')
#plt.xlim([-17,-12])
#plt.ylim([1e-6, 8])
#plt.yscale("log")
#plt.legend()
axes.set_xscale('log')
plt.tight_layout()
plt.savefig(output_directory + 'freesp_be.pdf')
plt.close()

import ipdb; ipdb.set_trace()
