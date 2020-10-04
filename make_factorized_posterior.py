import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import pickle
import numpy as np
from scipy.integrate import trapz
from sklearn.neighbors import KernelDensity as KD

from enterprise_warp.results import EnterpriseWarpResult
from enterprise_warp.results import parse_commandline

class FactorizedPosteriorResult(EnterpriseWarpResult):
  def __init__(self, opts):
    super(FactorizedPosteriorResult, self).__init__(opts)
    self.kde = dict()
    self.log_prob = dict()
    self.log_prob_factorized = None
    self.saved_factorized_kde_name = self.outdir_all + 'factorized_kde.pkl'

  def main_pipeline(self, prior, prior_type='uniform', inc_psrs=None):
    self.collect_kde(inc_psrs=inc_psrs)
    if prior_type=='uniform':
      self.x_vals = np.linspace(prior[0], prior[1], 1000)[:, np.newaxis]
    self.calculate_factorized_posterior()
    self.plot_results()

  def collect_kde(self, preload=True, inc_psrs=None):

    if preload and os.path.isfile(self.saved_factorized_kde_name):
      with open(self.saved_factorized_kde_name, 'rb') as handle:
        self.kde = pickle.load(handle)
    else:
      for psr_dir in self.psr_dirs:
 
        if inc_psrs is not None and psr_dir not in inc_psrs:
          continue
 
        self.psr_dir = psr_dir
  
        success = self._scan_psr_output()
        if not success:
          continue
  
        success = self.load_chains()
        if not success:
          continue
        if len(self.unique) > 1:
          raise ValueError('PTMCMC must sample only one model (parameter \
                            estimation, not model selection).')
  
        self._get_par_mask()
        if np.sum(self.par_mask) != 1:
          message = 'Here, --par must correspond only to one parameter. \
                     Current --par: ' + ','.join(opts.par)
          raise ValueError(message)
  
        self.get_distribution_kde()

      self.save_kdes()

  def get_distribution_kde(self, bandwidth=0.75):
    """ Currently works only with 1D distributions """
    dist_values = self.chain_burn[:,self.par_mask]
    kernel = KD(kernel='gaussian', bandwidth=bandwidth)
    self.kde[self.psr_dir] = kernel.fit(dist_values)

  def save_kdes(self):
    with open(self.saved_factorized_kde_name, 'wb') as handle:
      pickle.dump(self.kde, handle, protocol=pickle.HIGHEST_PROTOCOL)

  def calculate_factorized_posterior(self):
    self.log_prob_factorized = 0
    print('Calculating posterior densities on a grid...')
    for psr_dir, kde in self.kde.items():
      self.log_prob[psr_dir] = kde.score_samples(self.x_vals)
      self.log_prob_factorized += self.log_prob[psr_dir]
    self.prob_factorized_norm = np.exp(self.log_prob_factorized) / \
                                trapz(np.exp(self.log_prob_factorized), \
                                x=self.x_vals.T) 

  def plot_results(self):
    for psr_dir in self.kde.keys():
      plt.plot(self.x_vals, np.exp(self.log_prob[psr_dir]), alpha=0.1,
               color='grey', linewidth=2)
    plt.plot(self.x_vals, self.prob_factorized_norm, color='red',
             label='Factorized posterior')
    plt.legend()
    plt.xlabel('$log_{10}A_{CRN}$')
    plt.ylabel('Factorized posterior probability')
    plt.tight_layout()
    plt.savefig(self.outdir_all + 'factorized_posterior.png')
    plt.close()


opts = parse_commandline()

result_obj = FactorizedPosteriorResult(opts)
#inc_psrs = ['20_J1909-3744','0_J0437-4715','24_J2145-0750','11_J1603-7202','5_J1024-0719','23_J2129-5721','22_J2124-3358','10_J1600-3053']
result_obj.main_pipeline([-20., -6.])#, inc_psrs=inc_psrs)
