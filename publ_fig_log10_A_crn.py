import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from enterprise_warp.results import parse_commandline
from enterprise_warp.results import EnterpriseWarpResult
from make_factorized_posterior import FactorizedPosteriorResult

# What results to grab
#psrs_set = '/home/bgonchar/correlated_noise_pta_2020/params/pulsar_set_x_1.dat'
psrs_set = '/home/bgonchar/correlated_noise_pta_2020/params/pulsar_set_all.dat'
output_directory = '/home/bgonchar/correlated_noise_pta_2020/publication_figures/'
result = [
#'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_common_pl_factorized_20200915.dat',
'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_snall_wnfix_pe_common_pl_factorized_30_nf_20210126.dat',

#'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_ms_common_pl_vargam_set_x_1_ephem_0_20200928.dat',
#'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_ms_common_pl_set_x_1_ephem_0_20201103.dat'
'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_snall_ms_common_pl_fixgam_30_nf_set_all_ephem_0_20210125.dat',
]
par = [
['gw'],
#['gw_log10_A'],
['gw','nmodel'],
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
ngAmin, ngAmax, ngAmed = np.log10(1.4e-15), np.log10(2.7e-15), np.log10(1.9e-15)
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif",
  #"font.serif": ["Palatino"],
})
font = {'family' : 'serif',
        'size'   : 17}
fig, axes = plt.subplots()
axes.axvline(ngAmed, color='#FFB300')
for rr, pp, nm, ll in zip(result, par, nmodel, labels):
  opts.__dict__['result'] = rr
  opts.__dict__['par'] = pp
  if 'factorized' in rr:
    result_obj = FactorizedPosteriorResult(opts, psrs_set=psrs_set)
    result_obj.main_pipeline([-20., -6.], plot_results = False, plot_psrs=False)
    fobj = result_obj.make_figure(axes, label = ll, colorpsr='#9E9E9E', colorall='#1976D2', lwpsr=0.5, lwall=2, alphapsr=.5, alphaall=1.0)
  else:
    opts.__dict__['load_separated'] = True
    result_obj = EnterpriseWarpResult(opts)
    result_obj.main_pipeline()
    model_mask = np.round(result_obj.chain_burn[:,result_obj.ind_model]) == nm
    values = result_obj.chain_burn[model_mask,:]
    values = values[:,result_obj.par_mask]
    values = values[:,0]  # removing  nmodel samples (column 1)
    plt.hist(values, bins=20, density=True, histtype='stepfilled', alpha=0.9, facecolor='#4FC3F7', hatch='/', edgecolor='#1976D2', label = ll)
#plt.vlines(np.log10(1.9e-15), ymin, ymax, linestyles="dotted", colors="black", label="NANOGrav 12.5-yr CPL")
rect = patches.Rectangle((ngAmin, ymin), ngAmax-ngAmin, ymax-ymin, edgecolor=None, facecolor='#FFB300', alpha=0.6, label='NANOGrav')
axes.add_patch(rect)
axes.set_xlabel('$\\log_{10}A$', fontdict=font)
axes.set_ylabel('Posterior probability', fontdict=font)
plt.xlim([-17,-12])
plt.ylim([ymin, ymax])
plt.yscale("log")
#plt.legend()
axes.tick_params(axis='y', labelsize = font['size'])
axes.tick_params(axis='x', labelsize = font['size'])
plt.grid(b=None)
plt.tight_layout()
plt.savefig(output_directory + 'log10_A_crn.pdf')
plt.close()
