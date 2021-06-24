"""
Usage example:
python make_factorized_posterior.py --result /home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_wnfix_pe_common_pl_factorized_20200916.dat --par gw
Where gwb_log10_A is the parameter that we produce factorized posterior for.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import pickle
import numpy as np
from scipy.integrate import trapz
from sklearn.neighbors import KernelDensity as KD

from enterprise_extensions.model_utils import bayes_fac as sd_bf

from enterprise_warp.results import EnterpriseWarpResult
from enterprise_warp.results import parse_commandline

class FactorizedPosteriorResult(EnterpriseWarpResult):
  def __init__(self, opts, psrs_set=''):
    super(FactorizedPosteriorResult, self).__init__(opts)
    self.kde = dict()
    self.posterior = dict()
    self.log_prob = dict()
    self.log_prob_factorized = None
    self.sd_bf = dict()
    self.dropout = dict()
    self.interpret_psrs_set(psrs_set)
    self.saved_factorized_kde_name = self.outdir_all + 'factorized_kde' + \
                                     self.psrs_set_name + '.pkl'
    self.saved_factorized_posterior_name = self.outdir_all + \
                                           'factorized_posterior' + \
                                           self.psrs_set_name + '.pkl'

  def interpret_psrs_set(self, psrs_set):
    """
    Pulsar set name format: /path/pulsar_set_NAME.dat
    It must contain a list of pulsars that are allowed to be used.
    """
    if os.path.isfile(psrs_set):
      self.psrs_set_name = os.path.basename(psrs_set)
      self.psrs_set_name = self.psrs_set_name.split('pulsar_set_')[1]
      self.psrs_set_name = '_' + self.psrs_set_name.split('.dat')[0]
      self.allowed_psrs = np.loadtxt(psrs_set, dtype=np.unicode_)
    else:
      self.allowed_psrs = []
      self.psrs_set_name = ''

  def main_pipeline(self, prior, prior_type='uniform', inc_psrs=None,
                    plot_results=True, plot_psrs=True):
    if prior_type=='uniform' and np.array(prior).shape==(2,):
      self.x_vals = np.linspace(prior[0], prior[1], 1000)[:, np.newaxis]
    elif prior_type=='uniform' and np.array(prior).shape!=(2,):
      x_val_list = (np.linspace(pr[0], pr[1], 1000) for pr in prior)
      mg = np.meshgrid(*x_val_list)
      self.x_vals = np.vstack((mgx.flatten() for mgx in mg)).T

    self.collect_kde(inc_psrs=inc_psrs)
    self.calculate_factorized_posterior()

    if plot_results:
      self.plot_results()

    if plot_psrs:
      self.plot_psrs()

  def collect_kde(self, preload=True, inc_psrs=None):

    if preload and os.path.isfile(self.saved_factorized_kde_name):
      with open(self.saved_factorized_kde_name, 'rb') as handle:
        saved_data = pickle.load(handle)
      self.kde = saved_data['kde']
      self.sd_bf = saved_data['sd_bf']
      print('Pre-loaded KDEs')
    else:
      print('Calculating KDEs')
      for psr_dir in self.psr_dirs:

        if psr_dir.split('_')[1] not in self.allowed_psrs:
          print('Skipping ', psr_dir, ', it is not in set ', self.psrs_set_name)
          continue
 
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
        #if np.sum(self.par_mask) != 1:
        #  message = 'Here, --par must correspond only to one parameter. \
        #             Current --par: ' + ','.join(self.opts.par)
        #  raise ValueError(message)
  
        self.get_distribution_kde()
        self.get_sd_bayes_factor()

      self.save_kdes()

  def get_distribution_kde(self, bandwidth=0.05):
    """ Currently works only with 1D distributions """
    dist_values = self.chain_burn[:,self.par_mask]
    kernel = KD(kernel='gaussian', bandwidth=bandwidth)
    self.kde[self.psr_dir] = kernel.fit(dist_values)

  def get_sd_bayes_factor(self):
    """ Savage-Dickey Bayes Factor for parameter --par (i.e., log10_A_gw) """
    self.sd_bf[self.psr_dir] = sd_bf(self.chain_burn[:,self.par_mask], \
                                     logAmin=np.min(self.x_vals), \
                                     logAmax=np.max(self.x_vals))

  def save_kdes(self):
    output = dict()
    output['kde'] = self.kde
    output['sd_bf'] = self.sd_bf
    with open(self.saved_factorized_kde_name, 'wb') as handle:
      pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

  def save_posterior(self):
    output_posterior = {
      "prob_factorized_norm": self.prob_factorized_norm,
      "log_prob": self.log_prob,
      "x_vals": self.x_vals
    }
    with open(self.saved_factorized_posterior_name, 'wb') as handle:
      pickle.dump(output_posterior, handle, protocol=pickle.HIGHEST_PROTOCOL)

  def calculate_factorized_posterior(self, preload=True):
    if preload and os.path.isfile(self.saved_factorized_posterior_name):
      with open(self.saved_factorized_posterior_name, 'rb') as handle:
         posterior = pickle.load(handle)
      self.__dict__.update(posterior)
      print('Pre-loaded factorized posterior')
    else:
      self.log_prob_factorized = 0
      print('Calculating posterior densities on a grid...')
      for psr_dir, kde in self.kde.items():
        print(os.path.basename(psr_dir))
        self.log_prob[psr_dir] = kde.score_samples(self.x_vals)
        self.log_prob_factorized += self.log_prob[psr_dir]
      self.prob_factorized_norm = np.exp(self.log_prob_factorized) / \
                                  trapz(np.exp(self.log_prob_factorized), \
                                  x=self.x_vals.T)
      self.save_posterior()

  def make_figure(self, ax, label='Factorized posterior', colorpsr='grey', 
                  colorall='dodgerblue', lwpsr=1, lwall=2, alphapsr=0.1,
                  alphaall=1.0, only_total=False):
    if not only_total:
      for psr_dir in self.kde.keys():
        fobj = ax.plot(self.x_vals, np.exp(self.log_prob[psr_dir]),
                       alpha=alphapsr, color=colorpsr, linewidth=lwpsr)
      fobj += ax.plot(self.x_vals, self.prob_factorized_norm, color=colorall,
                      alpha=alphaall, linewidth=lwall, label=label)
    else:
      fobj = ax.plot(self.x_vals, self.prob_factorized_norm, color=colorall,
                     alpha=alphaall, linewidth=lwall, label=label)
    return fobj

  def plot_psrs(self, alphapsr=1., colorpsr='r', lwpsr=1., ymin=1e-12, ymax=8.):
    for psr_dir in self.kde.keys():
      plt.plot(self.x_vals, np.exp(self.log_prob[psr_dir]),
               alpha=alphapsr, color=colorpsr, linewidth=lwpsr, label=psr_dir)
      plt.xlim([-17,-12])
      plt.ylim([ymin, ymax])
      plt.yscale("log")
      plt.legend()
      plt.xlabel('$\log_{10}A$')
      plt.ylabel('Posterior probability')
      plt.tight_layout()
      plt.savefig(self.outdir_all + 'kde_posterior_gw_' + \
                  psr_dir + '.png')
      plt.close()

  def plot_results(self):
    fig, axes = plt.subplots()
    self.make_figure(axes)
    #for psr_dir in self.kde.keys():
    #  plt.plot(self.x_vals, np.exp(self.log_prob[psr_dir]), alpha=0.1,
    #           color='grey', linewidth=2)
    #plt.plot(self.x_vals, self.prob_factorized_norm, color='red',
    #         label='Factorized posterior')
    plt.legend()
    plt.xlabel('$\log_{10}A_{CRN}$')
    plt.ylabel('Factorized posterior probability')
    #plt.xlim([-15.5, -13.5])
    plt.tight_layout()
    plt.savefig(self.outdir_all + 'factorized_posterior' + \
                self.psrs_set_name + '.png')
    plt.close()

def main():
  """
  The pipeline script
  """
  opts = parse_commandline()
  psrs_set = '/home/bgonchar/correlated_noise_pta_2020/params/pulsar_set_all.dat'
  result_obj = FactorizedPosteriorResult(opts, psrs_set=psrs_set)
  #inc_psrs = ['20_J1909-3744','0_J0437-4715','24_J2145-0750','11_J1603-7202','5_J1024-0719','23_J2129-5721','22_J2124-3358','10_J1600-3053']
  result_obj.main_pipeline([-20., -6.])#, inc_psrs=inc_psrs)

if __name__=='__main__':
  main()
