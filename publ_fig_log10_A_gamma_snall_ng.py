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
'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_snall_pe_common_pl_vargam_30_nf_set_all_ephem_0_20210125.dat',
'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_common_pl_vargam_5_nf_set_all_ephem_0_20201217.dat',
'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_common_pl_vargam_20_nf_set_all_ephem_0_20201217.dat',
'/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_common_pl_vargam_30_nf_set_all_ephem_0_20201217.dat',
]
par = [
['gw','nmodel'],
['gw'],
['gw'],
['gw'],
]
nmodel = [
0,
0,
0,
0,
]
labels = [
"$~$", #"$n_\\mathrm{c} = 5$",
"$~$", #"$n_\\mathrm{c} = 5$",
"$~$", #"$n_\\mathrm{c} = 20$",
"$~$", #"$n_\\mathrm{c} = 30$",
]
linestyles = [
"-",
":",
"--",
"-",
"-.", # NG
"-.", # EPTA
]
shade = [
True,
False,
False,
True,
False, # NG
False, # EPTA
]
shade_alpha = [
0.4,
0.0,
0.0,
0.5,
0.0, # NG
0.0, # EPTA
]
linewidths = [
1.0,
0.5,
0.5,
1.0,
1.0, #NG
1.0, #NG
]
colors = [
"grey",
"#1976D2",
"#1976D2",
"#1976D2",
"orange", #"#FFB300", #NG
"#27AB37", #"#FFB300", #EPTA
]
extents=[[2.0, 7.0], [-16.0, -13.8]]

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
  if values.shape[1]==3:
    values = values[:,:-1] # to eliminate nmodel
  cc.add_chain(values, parameters=["$\\gamma$", "$\\log_{10}A$"], name=ll)

# Adding NANOGrav
dirr = '/fred/oz002/rshannon/ng_compare/'
ngchain = np.loadtxt(dirr+'ng.dat')
cc.add_chain(ngchain[:,1:], parameters=["$\\gamma$","$\\log_{10}A$"], name="NANOGrav")

# Adding EPTA
dirr = '/fred/oz031/pta_data/chains_epta_dr2/'
eptachain = np.loadtxt(dirr+'chain_1.txt')[:,-6:-4]
cc.add_chain(eptachain, parameters=["$\\gamma$","$\\log_{10}A$"], name="EPTA")

cc.configure(summary=False, linestyles=linestyles, linewidths=linewidths,
             shade=shade, bar_shade=shade, shade_alpha=shade_alpha, serif=True,
             legend_color_text=False, legend_artists=False, colors=colors)
             # legend_kwargs={"loc": "best"}, legend_location=(0, 0)
plt.rcParams['axes.grid'] = False
cfig = cc.plotter.plot(extents=extents, filename=output_directory+'log10_A_gamma_snall_ng_epta.pdf', truth={"$\\gamma$": 13./3.})

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
