import numpy as np
from scipy.integrate import trapz
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
                              log10A_min = -20., log10A_max = -6.):
  dropout = dict()
  for psr in log_prob_k.keys():
    #idx_psr = list(self.log_prob.keys()).index(psr_dir)
    #log_prob_all = list(self.log_prob.values())
    #log_prob_other_psrs = log_prob_all[:idx_psr] + log_prob_all[idx_psr+1:]
    #log_prob_other_psrs = np.sum(log_prob_other_psrs, axis=0)
    prior = np.repeat(1/(log10A_max - log10A_min), len(x_vals))
    integrand = np.exp(log_prob_k[psr] + log_prob_jnek[psr]) / prior
    dropout[psr] = trapz(integrand, x=x_vals.T)
  return dropout

def main():
  """
  The pipeline script
  """
  opts = parse_commandline()
  output_directory = '/home/bgonchar/correlated_noise_pta_2020/publication_figures/'
  psrs_set = '/home/bgonchar/correlated_noise_pta_2020/params/pulsar_set_x_1.dat'

  # Factorized results per pulsar
  opts.result = '/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_wnfix_pe_common_pl_factorized_20200916.dat'
  result_obj = FactorizedPosteriorResult(opts, psrs_set=psrs_set)
  result_obj.main_pipeline([-20., -6.])
  bf_k = result_obj.sd_bf
  x_vals = result_obj.x_vals
  log_prob_k = result_obj.log_prob

  # Factorized results for all PSRs in set except each one, one by one
  opts.result = '/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_pe_cpl_set_x_1_factorized_dropout_20201105.dat'
  result_obj = FactorizedPosteriorResult(opts, psrs_set=psrs_set)
  result_obj.main_pipeline([-20., -6.])
  log_prob_jnek = result_obj.log_prob
  #if not x_vals == result_obj.x_vals:
  #  raise ValueError('Different x-vals')

  dropout = calculate_dropout_factors(x_vals, log_prob_k, log_prob_jnek, bf_k)
  dropout_keys = list()
  dropout_vals = list()
  for kk, vv in sorted(dropout.items(), key=lambda x: x[1], reverse=True):
    dropout_vals.append(vv[0])
    dropout_keys.append(kk.split('_')[1])
  #dropout_vals = [dd[0] for dd in dropout.values()]
  #dropout_keys = [kk for kk in dropout.keys()]

  dummy_xvals = np.linspace(0,1, len(dropout_vals))
  #fig = plt.figure()
  #ax1 = fig.add_subplot(111)
  plt.scatter(dummy_xvals, dropout_vals, s=5, c="green")
  plt.xticks(dummy_xvals, dropout_keys, rotation='vertical')
  #ax1.set_xticklabels(dropout_vals)
  plt.yscale("log")
  plt.xlabel("PSR")
  plt.ylabel("Dropout factor")
  plt.tight_layout()
  plt.savefig(output_directory + 'dropout.pdf')
  plt.close()

  import ipdb; ipdb.set_trace()

if __name__=='__main__':
  main()
