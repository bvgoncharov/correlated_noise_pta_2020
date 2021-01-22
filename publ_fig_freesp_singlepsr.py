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
'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_freesp_30_nf_singlepsr_20210120.dat',
'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_freesp_30_nf_singlepsr_20210120.dat',
'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_freesp_30_nf_singlepsr_20210120.dat',
'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_freesp_30_nf_singlepsr_20210120.dat',
'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_freesp_30_nf_singlepsr_20210120.dat',
]
names = [
"J1022+1001",
"J1600-3053",
"J2145-0750",
"J2241-5236",
"J1909-3744",
]
dfs = [
2.2315554798611935e-09,
2.2315554798611935e-09,
2.2473821144701385e-09,
3.8644009529303604e-09,
2.2315554798611935e-09,
]
par = [
['gw_log10_rho'],
['gw_log10_rho'],
['gw_log10_rho'],
['gw_log10_rho'],
['gw_log10_rho'],
]
nmodel = [
0,
0,
0,
0,
0,
]
labels = [
None,
None,
None,
None,
None,
]
facecolors = [
"none",
"none",
"none",
"none",
"none",
]
edgecolors = [
"red",
"blue",
"black",
"green",
"orange"
]
linewidths = [
0.5,
0.5,
0.5,
0.5,
0.5,
]
alphas = [
0.5,
0.5,
0.5,
0.5,
0.5,
]
angles = [
None,
None,
None,
None,
None,
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
for rr, pp, nm, ll, fc, ec, lw, al, aa, df, na in zip(result, par, nmodel, labels, facecolors, edgecolors, linewidths, alphas, angles, dfs, names):
  opts.__dict__['result'] = rr
  opts.__dict__['par'] = pp
  opts.__dict__['name'] = na

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
  fobj = result_obj.make_figure(axes, label=ll, facecolor=fc, edgecolor=ec, alpha=al, linewidth=lw, df=df, yval="log10_psd", bw_method=0.3, widthdf = 0.3)
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
plt.savefig(output_directory + 'freesp_singlepsr.pdf')
plt.close()

#import ipdb; ipdb.set_trace()
