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
'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_gwb_fixslope_interp_orf_20_nf_set_x_1_ephem_0_20201111.dat'
]
par = [
['corr_coeff'],
]
nmodel = [
0,
]
labels = [
"ORF",
]
colors = [
"purple",
#"#F39C12",
#"#E39C12"
]
angles = [
np.linspace(0, np.pi,7),
]


opts = parse_commandline()
opts.__dict__['logbf'] = True

#plt.style.use('seaborn-white')
fig, axes = plt.subplots()
hd_xvals = np.linspace(0.000001,np.pi,100)
hd_yvals = hd_orf(hd_xvals)
hd_xvals = np.append(0,hd_xvals)
hd_yvals = np.append(1,hd_yvals)
plt.plot(hd_xvals,hd_yvals,color="#F39C12", linestyle="--")
for rr, pp, nm, ll, cc, aa in zip(result, par, nmodel, labels, colors, angles):
  opts.__dict__['result'] = rr
  opts.__dict__['par'] = pp

  result_obj = EnterpriseWarpResult(opts)
  result_obj.main_pipeline()
  model_mask = np.round(result_obj.chain_burn[:,result_obj.ind_model]) == nm
  values = result_obj.chain_burn[model_mask,:]
  values = values[:,result_obj.par_mask]
  axes.violinplot(values, positions=aa, widths=np.sum(aa)/len(aa)/3, showextrema=False)
  #plt.hist(values, bins=20, density=True, histtype='stepfilled', alpha=0.5, facecolor=cc, hatch=hh, edgecolor=cc, label = ll)
plt.xticks(ticks=[0,np.pi/4,np.pi/2,3*np.pi/4,np.pi], labels=["$0$","$\\frac{\\pi}{4}$","$\\frac{\\pi}{2}$","$\\frac{3\\pi}{4}$","$\\pi$"])
plt.ylabel('$\\mathrm{Spatial~correlation,~}\\xi(\\zeta)$')
plt.xlabel('$\\mathrm{Angle~between~Earth-pulsar~baselines,~} \\zeta \\mathrm{~[rad]}$')
#plt.xlim([-17,-12])
#plt.ylim([1e-6, 8])
#plt.yscale("log")
#plt.legend()
plt.tight_layout()
plt.savefig(output_directory + 'orf.pdf')
plt.close()

import ipdb; ipdb.set_trace()
