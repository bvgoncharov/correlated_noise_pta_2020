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
'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_cpl_gwb_noauto_fixslope_30_nf_set_all_ephem_0_20201219.dat',
'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_cpl_gwb_noauto_fixslope_30_nf_set_all_ephem_0_20201219.dat',
#'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_ms_common_pl_set_x_1_ephem_0_20201103.dat'
]
par = [
['gw_log10_A_hd'],
['gw_log10_A'],
['gw_log10_A_hd'],
['gw_log10_A'],
]
nmodel = [
0,
0,
0,
0,
#1
]
labels = [
"HD no-auto",
"CPL",
"HD no-auto",
"CPL",
#"CPL"
]
facecolor = [
"#E53935",
"#4FC3F7",
"None",
"None",
#"#E39C12"
]
edgecolor = [
"None",
"None",
"#795548",
"#1976D2",
]
bins = [
60,
10,
60,
10,
#"/"
]
linewidth = [
0,
0,
2,
2,
]
alpha = [
0.5,
0.5,
1,
1,
]

opts = parse_commandline()
opts.__dict__['logbf'] = True
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif",
  #"font.serif": ["Palatino"],
})
#plt.style.use('seaborn-white')
fig, axes = plt.subplots()
for rr, pp, nm, ll, fc, ec, bb, lw, al in zip(result, par, nmodel, labels, facecolor, edgecolor, bins, linewidth, alpha):
  opts.__dict__['result'] = rr
  opts.__dict__['par'] = pp

  result_obj = EnterpriseWarpResult(opts)
  result_obj.main_pipeline()
  model_mask = np.round(result_obj.chain_burn[:,result_obj.ind_model]) == nm
  values = result_obj.chain_burn[model_mask,:]
  values = values[:,result_obj.par_mask]
  # Special case - remove log10A_hd column:
  if values.shape[1]==2:
    values = values[:,0]
  plt.hist(values, bins=bb, density=True, histtype='stepfilled', alpha=al, hatch=None, facecolor=fc, edgecolor=ec, label = ll, linewidth=lw)
plt.xlabel('$\log_{10}A$')
plt.ylabel('Posterior probability')
#plt.xlim([-17,-12])
#plt.ylim([1e-6, 8])
plt.yscale("log")
#plt.legend()
plt.grid(b=None)
plt.tight_layout()
plt.savefig(output_directory + 'log10_A_hd.pdf')
plt.close()
