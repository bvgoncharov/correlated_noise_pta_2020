import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

import enterprise.constants as const
from enterprise_warp.results import parse_commandline
from enterprise_warp.results import EnterpriseWarpResult
from make_freespec_violin import ViolinFreespecResult

def powerlaw_psd(freqs,A,gamma):
  return A**2/12./np.pi**2 * (freqs*const.yr)**(-gamma) * const.yr**3

# What results to grab
output_directory = '/home/bgonchar/correlated_noise_pta_2020/publication_figures/'
result = [
'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_cpl_freesp_30_nf_set_all_ephem_0_20201222.dat',
'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_cpl_freesp_30_nf_set_x_1_ephem_0_20201015.dat',
]
par = [
['gw_log10_rho'],
['gw_log10_rho'],
]
nmodel = [
0,
0,
]
labels = [
None,
None,
]
facecolors = [
"white",
"#4FC3F7",
#"#E39C12",
]
edgecolors = [
"#1976D2",
"#4FC3F7",
]
linewidths = [
2,
0,
]
alphas = [
1.0,
0.5
]
angles = [
None,
None,
]
dfs = [
2.11253918760193e-09,
2.2292808626385243e-09,
]


opts = parse_commandline()
opts.__dict__['logbf'] = True

#plt.style.use('seaborn-white')
fig = plt.figure()
axes = fig.add_subplot(111)
font = {'family' : 'serif',
        'size'   : 12}
#plt.rcParams.update({
#  "text.usetex": True,
#  "text.latex.unicode": True,
#  "font.family": "serif",
#  #"font.serif": ["Palatino"],
#})
count_order = 0
for rr, pp, nm, ll, fc, ec, lw, al, aa, df in zip(result, par, nmodel, labels, facecolors, edgecolors, linewidths, alphas, angles, dfs):
  opts.__dict__['result'] = rr
  opts.__dict__['par'] = pp

  result_obj = ViolinFreespecResult(opts)
  result_obj.main_pipeline(plot=False)
  if count_order==0:
    axes.axvline(x=result_obj.inv_earth_orb_period, ls=':', c='grey')
    plt.text(result_obj.inv_earth_orb_period, -4.5, 'Earth', fontdict=font, color='grey', rotation=90)
    axes.axvline(x=result_obj.inv_ven_orb_period, ls=':', c='grey')
    plt.text(result_obj.inv_ven_orb_period, -4.5, 'Venus', fontdict=font, color='grey', rotation=90)
    axes.axvline(x=result_obj.inv_mar_orb_period, ls=':', c='grey')
    plt.text(result_obj.inv_mar_orb_period, -4.5, 'Mars', fontdict=font, color='grey', rotation=90)
    axes.axvline(x=result_obj.inv_sat_orb_period, ls=':', c='grey')
    plt.text(result_obj.inv_jup_orb_period, -10.5, 'Jupiter', fontdict=font, color='grey', rotation=90)
    axes.axvline(x=result_obj.inv_jup_orb_period, ls=':', c='grey')
    plt.text(result_obj.inv_sat_orb_period, -10.5, 'Saturn', fontdict=font, color='grey', rotation=90)
  fobj = result_obj.make_figure(axes, label=ll, facecolor=fc, edgecolor=ec, alpha=al, linewidth=lw, df=df, yval="log10_psd", bw_method=0.3)
  count_order += 1
  #model_mask = np.round(result_obj.chain_burn[:,result_obj.ind_model]) == nm
  #values = result_obj.chain_burn[model_mask,:]
  #values = values[:,result_obj.par_mask]
  #axes.violinplot(values, positions=aa, widths=np.sum(aa)/len(aa)/3, showextrema=False)
  #plt.hist(values, bins=20, density=True, histtype='stepfilled', alpha=0.5, facecolor=cc, hatch=hh, edgecolor=cc, label = ll)

#plt.ylabel('$\\log_{10}(\\rho\\mathrm{[s]})$')
plt.ylabel('$\\log_{10}(P\\mathrm{[s}^3\\mathrm{]})$')
plt.xlabel('$\\mathrm{Frequency,~[Hz]}$')
AA = 2.2e-15
gamma = 13./3.
xf = np.linspace(1e-9,7e-8,10)
#plt.plot(xf, np.log10(np.sqrt(powerlaw_psd(xf,AA,gamma)*df)))
plt.plot(xf, np.log10(powerlaw_psd(xf,AA,gamma)), color='black')
#axes.set_xlabel('$\\mathrm{Frequency~[Hz]}$', fontdict=font)
#axes.set_ylabel('$\\log_{10}(\\rho\\mathrm{[s]})$', fontdict=font)
plt.xlim([xf[0],xf[-1]])
#plt.ylim([1e-6, 8])
#plt.yscale("log")
#plt.legend()
axes.set_xscale('log')
#axes.tick_params(axis='y', labelsize = font['size'])
#axes.tick_params(axis='x', labelsize = font['size'])
plt.grid(b=None)
plt.tight_layout()
plt.savefig(output_directory + 'freesp.pdf')
plt.close()

#import ipdb; ipdb.set_trace()
