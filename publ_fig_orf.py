import numpy as np
from matplotlib import pyplot as plt

from enterprise_warp.results import parse_commandline
from enterprise_warp.results import EnterpriseWarpResult
from make_factorized_posterior import FactorizedPosteriorResult

def hd_orf(eta):
  """ Angle eta in rad """
  comb = (1. - np.cos(eta))/2.
  return 0.5 - 0.25*comb + 1.5*comb*np.log(comb)

# What results to grab
output_directory = '/home/bgonchar/correlated_noise_pta_2020/publication_figures/'
result = [
'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_gwb_fixslope_interp_orf_30_nf_set_all_ephem_0_20201218.dat',
'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_gwb_fixslope_interp_orf_set_x_1_ephem_0_20201105.dat',
#'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_gwb_fixslope_interp_orf_20_nf_set_x_1_skyscr_1_20201202.dat',
]
par = [
['corr_coeff'],
['corr_coeff'],
]
nmodel = [
0,
0,
]
labels = [
"ORF",
"x1"
#"Sky-scrambled",
]
colors = [
"white",
"#E53935",
#"#F39C12",
#"#E39C12"
]
edgecolors = [
"#795548",
"#E53935",
]
angles = [
np.linspace(0, np.pi,7),
np.linspace(0, np.pi,7),
]
linewidths = [
2,
0,
]
alphas = [
1.0,
0.5
]


opts = parse_commandline()
opts.__dict__['logbf'] = True
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif",
  #"font.serif": ["Palatino"],
})
font = {'family' : 'serif',
        'size'   : 12}
#plt.style.use('seaborn-white')
fig = plt.figure()
axes = fig.add_subplot(111)
hd_xvals = np.linspace(0.000001,np.pi,100)
hd_yvals = hd_orf(hd_xvals)
#hd_xvals = np.append(0,hd_xvals)
#hd_yvals = np.append(1,hd_yvals)
plt.plot(hd_xvals,hd_yvals, color="black", linestyle="--")
for rr, pp, nm, ll, cc, ec, aa, lw, al in zip(result, par, nmodel, labels, colors, edgecolors, angles, linewidths, alphas):
  opts.__dict__['result'] = rr
  opts.__dict__['par'] = pp

  result_obj = EnterpriseWarpResult(opts)
  result_obj.main_pipeline()
  model_mask = np.round(result_obj.chain_burn[:,result_obj.ind_model]) == nm
  values = result_obj.chain_burn[model_mask,:]
  values = values[:,result_obj.par_mask]
  fobj = axes.violinplot(values, positions=aa, widths=np.sum(aa)/len(aa)/3, \
                         showextrema=False, bw_method=0.3)
  for pc in fobj['bodies']:
    pc.set_facecolor(cc)
    pc.set_edgecolor(ec)
    pc.set_linewidth(lw)
    pc.set_alpha(al)
  #plt.hist(values, bins=20, density=True, histtype='stepfilled', alpha=0.5, facecolor=cc, hatch=hh, edgecolor=cc, label = ll)
#plt.xticks(ticks=[0,np.pi/4,np.pi/2,3*np.pi/4,np.pi], labels=["$0$","$\\frac{\\pi}{4}$","$\\frac{\\pi}{2}$","$\\frac{3\\pi}{4}$","$\\pi$"])
plt.xticks(ticks=[0,np.pi/6,2*np.pi/6,3*np.pi/6,4*np.pi/6,5*np.pi/6,6*np.pi/6], labels=["$0$","$30$","$60$","$90$","$120$","$150$","$180$"])
#plt.ylabel('$\\mathrm{Spatial~correlation,~}\\Gamma(\\zeta)$')
#plt.xlabel('$\\mathrm{Angle~between~Earth-pulsar~baselines,~} \\zeta \\mathrm{~[rad]}$')
axes.set_xlabel('$\\mathrm{Angle~between~Earth-pulsar~baselines,~} \\zeta \\mathrm{~[deg]}$', fontdict=font)
axes.set_ylabel('$\\mathrm{Spatial~correlation,~}\\Gamma(\\zeta)$', fontdict=font)
#plt.xlim([-17,-12])
plt.ylim([-1, 1])
#plt.yscale("log")
#plt.legend()
axes.tick_params(axis='y', labelsize = font['size'])
axes.tick_params(axis='x', labelsize = font['size'])
plt.grid(b=None)
plt.tight_layout()
plt.savefig(output_directory + 'orf.pdf')
plt.close()

import ipdb; ipdb.set_trace()
