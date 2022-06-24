import numpy as np
from scipy.integrate import trapz, simps
from matplotlib import pyplot as plt

from enterprise_warp.results import parse_commandline

from make_factorized_posterior import FactorizedPosteriorResult

#class DropoutResult(FactorizedPosteriorResult):
#  def __init__(self, opts, psrs_set=''):
#    super(DropoutResult, self).__init__(opts)
#
#  def calculate_dropout_factors(self):
#    """
#    Equation 7 in arXiv 2009.04496 (2020).
#    """
#    for psr_dir, log_prob in self.log_prob.items():
#      idx_psr = list(self.log_prob.keys()).index(psr_dir)
#      log_prob_all = list(self.log_prob.values())
#      log_prob_other_psrs = log_prob_all[:idx_psr] + log_prob_all[idx_psr+1:]
#      log_prob_other_psrs = np.sum(log_prob_other_psrs, axis=0)
#      import ipdb; ipdb.set_trace()
#      self.dropout[psr_dir] = self.sd_bf[psr_dir]

def calculate_dropout_factors(x_vals, log_prob_k, log_prob_jnek, bf_k,
                              return_integrand=False):
  dropout = dict()
  integrand = dict()
  log10A_min = np.min(x_vals)
  log10A_max = np.max(x_vals)
  for psr in log_prob_k.keys():
    #idx_psr = list(self.log_prob.keys()).index(psr_dir)
    #log_prob_all = list(self.log_prob.values())
    #log_prob_other_psrs = log_prob_all[:idx_psr] + log_prob_all[idx_psr+1:]
    #log_prob_other_psrs = np.sum(log_prob_other_psrs, axis=0)
    #import ipdb; ipdb.set_trace()
    prior = np.repeat(1/(log10A_max - log10A_min), len(x_vals))
    integrand[psr] = np.exp(log_prob_k[psr] + log_prob_jnek[psr]) / prior
    dropout[psr] = trapz(integrand[psr], x=x_vals.T)
    #dropout[psr] = simps(integrand, x=x_vals[:,0])
    dropout[psr] = dropout[psr] * bf_k[psr][0]
  if return_integrand:
    return dropout, integrand
  else:
    return dropout

def main():
  """
  The pipeline script
  """
  opts = parse_commandline()
  #output_directory = '/home/bgonchar/correlated_noise_pta_2020/publication_figures/'
  output_directory = '/fred/oz031/pta_gwb_priors_out/sim/20220303_efac_cp_factorz_220208_073559/'
  #psrs_set = '/home/bgonchar/correlated_noise_pta_2020/params/pulsar_set_x_1.dat'
  #psrs_set = '/home/bgonchar/correlated_noise_pta_2020/params/pulsar_set_all.dat'
  psrs_set = '/fred/oz031/pta_gwb_priors_out/sim_data/pulsar_set_all.dat'

  plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    #"font.serif": ["Palatino"],
  })
  font = {'family' : 'serif',
          'size'   : 17}

  # Factorized results per pulsar
  #opts.result = '/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_wnfix_pe_common_pl_factorized_20200916.dat'
  #opts.result ='/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_snall_wnfix_pe_common_pl_factorized_30_nf_20210126.dat'
  opts.result = '/home/bgonchar/pta_gwb_priors/params/factorized_one_sim_rnqcp_220208_073559_wnfix_pe_cpl_20220406.dat'
  result_obj = FactorizedPosteriorResult(opts, psrs_set=psrs_set)
  result_obj.main_pipeline([-20., -6.], plot_psrs=False)
  bf_k = result_obj.sd_bf
  x_vals = result_obj.x_vals
  log_prob_k = result_obj.log_prob

  # Factorized results for all PSRs in set except each one, one by one
  #opts.result = '/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_cpl_set_x_1_factorized_dropout_20201105.dat'
  #opts.result = '/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_snall_pe_cpl_30_nf_set_all_factorized_dropout_20210126.dat'
  opts.result = '/home/bgonchar/pta_gwb_priors/params/factorized_all_sim_rnqcp_220208_073559_pe_cpl_20220406.dat'
  result_obj = FactorizedPosteriorResult(opts, psrs_set=psrs_set)
  result_obj.main_pipeline([-20., -6.], plot_psrs=False)
  log_prob_jnek = result_obj.log_prob
  #if not x_vals == result_obj.x_vals:
  #  raise ValueError('Different x-vals')
  import ipdb; ipdb.set_trace()
  del log_prob_jnek['24_psr']
  del log_prob_jnek['2_psr'] 
  for key_false, key_true in zip(sorted(log_prob_jnek.keys()), sorted(log_prob_k.keys())): log_prob_jnek[key_true] = log_prob_jnek[key_false]

  dropout = calculate_dropout_factors(x_vals, log_prob_k, log_prob_jnek, bf_k)
  dropout_keys = list()
  dropout_vals = list()
  for kk, vv in sorted(dropout.items(), key=lambda x: x[1], reverse=True):
    dropout_vals.append(vv[0])
    dropout_keys.append(kk.split('_')[1].replace('-','--'))
  #dropout_vals = [dd[0] for dd in dropout.values()]
  #dropout_keys = [kk for kk in dropout.keys()]

  # And explanation for George and Dick for J2241's high contribution
  #sub_dropout_tot = {pp: np.array([]) for pp in dropout.keys()}
  #x_vals_plot = []
  #for ii in range(10,len(x_vals)+1, 10):
  #  sub_x_vals = x_vals[:ii,:]
  #  x_vals_plot.append(sub_x_vals[-1][0])
  #  sub_log_prob_k = dict()
  #  sub_log_prob_jnek = dict()
  #  for pp in log_prob_k.keys():
  #    sub_log_prob_k[pp] = log_prob_k[pp][:ii]
  #    sub_log_prob_jnek[pp] = log_prob_jnek[pp][:ii]
  #  sub_dropout = calculate_dropout_factors(sub_x_vals, sub_log_prob_k, sub_log_prob_jnek, bf_k)
  #  for pp in sub_dropout.keys():
  #    sub_dropout_tot[pp] = np.concatenate([sub_dropout_tot[pp],sub_dropout[pp]])
  #sub_x_vals_plot = np.arange(x_vals[10],x_vals[-1],x_vals[10]-x_vals[0])
  ##for pp in sub_dropout_tot.keys():
  #dpf, intg = calculate_dropout_factors(x_vals, log_prob_k, log_prob_jnek, bf_k, return_integrand=True)
  #psrs_test = ['0_J0437-4715','25_J2241-5236','20_J1909-3744','16_J1744-1134','13_J1713+0747','4_J1022+1001','10_J1600-3053','24_J2145-0750']
  #for pp in psrs_test:
  #  plt.plot(x_vals, intg[pp], label=pp.split('_')[1])
  #plt.legend()
  #plt.ylabel('Dropout factor integrand')
  #plt.xlabel('log10A')
  ##plt.ylim([0.01,6])
  #plt.xlim([-15.1, -14.4])
  #plt.savefig(output_directory + 'dropout_integrand.png')
  #plt.close()
  #for pp in psrs_test:
  #  plt.semilogy(x_vals_plot,sub_dropout_tot[pp], label=pp.split('_')[1])
  #plt.legend()
  #plt.ylabel('Dropout factor sub-integral')
  #plt.xlabel('log10A')
  #plt.ylim([0.01,6])
  #plt.savefig(output_directory + 'dropout_subintegral.png')
  #plt.xlim([-15.1, -14.4])
  #plt.savefig(output_directory + 'dropout_subintegral_zoom.png')
  #plt.close()
  #import ipdb; ipdb.set_trace()

  #OLD RESULTS - with no SN all

  # Factorized results per pulsar
  #opts.result = '/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_wnfix_pe_common_pl_factorized_20200916.dat'
  #opts.result ='/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_wnfix_pe_common_pl_factorized_30_nf_20201218.dat'
  #result_obj = FactorizedPosteriorResult(opts, psrs_set=psrs_set)
  #result_obj.main_pipeline([-20., -6.], plot_psrs=False)
  #bf_k = result_obj.sd_bf
  #x_vals = result_obj.x_vals
  #log_prob_k = result_obj.log_prob

  ## Factorized results for all PSRs in set except each one, one by one
  ##opts.result = '/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_cpl_set_x_1_factorized_dropout_20201105.dat'
  #opts.result = '/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_cpl_30_nf_set_all_factorized_dropout_20201219.dat'
  #result_obj = FactorizedPosteriorResult(opts, psrs_set=psrs_set)
  #result_obj.main_pipeline([-20., -6.], plot_psrs=False)
  #log_prob_jnek = result_obj.log_prob
  ##if not x_vals == result_obj.x_vals:
  ##  raise ValueError('Different x-vals')

  #dropout_old = calculate_dropout_factors(x_vals, log_prob_k, log_prob_jnek, bf_k)
  #dropout_keys_old = list()
  #dropout_vals_old = list()
  #for kk, vv in sorted(dropout.items(), key=lambda x: x[1], reverse=True):
  #  dropout_vals_old.append(dropout_old[kk][0])
  #  dropout_keys_old.append(kk.split('_')[1].replace('-','--'))
  #dropout_vals = [dd[0] for dd in dropout.values()]
  #dropout_keys = [kk for kk in dropout.keys()]

  dummy_xvals = np.linspace(0,1, len(dropout_vals))
  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  plt.grid(b=None, axis='y')
  plt.scatter(dummy_xvals, dropout_vals, s=25, c="#1976D2")
  #plt.scatter(dummy_xvals, dropout_vals_old, s=25, c="grey")
  plt.xticks(dummy_xvals, dropout_keys, rotation='vertical')
  #ax1.set_xticklabels(dropout_vals)
  plt.yscale("log")
  #plt.xlabel("PSR")
  ax1.set_xlabel('PSR', fontdict=font)
  #plt.ylabel("Dropout factor")
  ax1.set_ylabel('Dropout factor', fontdict=font)
  ax1.tick_params(axis='y', labelsize = font['size'])
  ax1.tick_params(axis='x', labelsize = font['size']-2)
  plt.tight_layout()
  plt.savefig(output_directory + 'dropout.pdf')
  plt.close()

  import ipdb; ipdb.set_trace()

if __name__=='__main__':
  main()
