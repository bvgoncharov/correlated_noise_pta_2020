import numpy as np
from matplotlib import pyplot as plt
from chainconsumer import ChainConsumer

from enterprise_warp.results import parse_commandline
from enterprise_warp.results import EnterpriseWarpResult
from make_factorized_posterior import FactorizedPosteriorResult

# What results to grab
#psrs_set = '/home/bgonchar/correlated_noise_pta_2020/params/pulsar_set_x_1.dat'
psrs_set = '/home/bgonchar/correlated_noise_pta_2020/params/pulsar_set_all.dat'
output_directory = '/home/bgonchar/correlated_noise_pta_2020/publication_figures/'
result = [
'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_common_pl_vargam_5_nf_set_all_ephem_0_20201217.dat',
'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_common_pl_vargam_20_nf_set_all_ephem_0_20201217.dat',
'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_common_pl_vargam_30_nf_set_all_ephem_0_20201217.dat',
]
par = [
['gw'],
['gw'],
['gw'],
]
nmodel = [
0,
0,
0,
]
labels = [
"$~$", #"$n_\\mathrm{c} = 5$",
"$~$", #"$n_\\mathrm{c} = 20$",
"$~$", #"$n_\\mathrm{c} = 30$",
]
linestyles = [
":",
"--",
"-",
]
shade = [
False,
False,
True,
]
shade_alpha = [
0.0,
0.0,
0.5,
]
linewidths = [
0.5,
0.5,
1.0,
]
colors = [
"#1976D2",
"#1976D2",
"#1976D2",
]
extents=[[2.0, 6.0], [-13.8, -15.4]]

opts = parse_commandline()
opts.__dict__['logbf'] = True

#plt.style.use('seaborn-white')

#fig, axes = plt.subplots()
cc = ChainConsumer()
for rr, pp, nm, ll in zip(result, par, nmodel, labels):
  opts.__dict__['result'] = rr
  opts.__dict__['par'] = pp
  opts.__dict__['load_separated'] = 1
  result_obj = EnterpriseWarpResult(opts)
  result_obj.main_pipeline()
  if result_obj.counts is not None:
    model_mask = np.round(result_obj.chain_burn[:,result_obj.ind_model]) == nm
    values = result_obj.chain_burn[model_mask,:]
    values = values[:,result_obj.par_mask]
  else:
    values = result_obj.chain_burn[:,result_obj.par_mask]
  cc.add_chain(values, parameters=["$\\gamma$", "$\\log_{10}A$"], name=ll)
cc.configure(summary=False, linestyles=linestyles, linewidths=linewidths,
             shade=shade, bar_shade=shade, shade_alpha=shade_alpha, serif=True,
             legend_color_text=False, legend_artists=False, colors=colors)
             # legend_kwargs={"loc": "best"}, legend_location=(0, 0)
plt.rcParams['axes.grid'] = False
cfig = cc.plotter.plot(extents=extents, filename=output_directory+'log10_A_gamma.pdf', truth={"$\\gamma$": 13./3.})

#plt.vlines(np.log10(1.9e-15), ymin, ymax, linestyles="dotted", colors="black", label="NANOGrav 12.5-yr CPL")
#plt.xlabel('$\log_{10}A$')
#plt.ylabel('Posterior probability')
#plt.xlim([-17,-12])
#plt.ylim([ymin, ymax])
#plt.yscale("log")
#plt.legend()
#plt.grid(b=None)
#plt.tight_layout()
#plt.savefig(output_directory + 'log10_A_gamma.pdf')
plt.close()
