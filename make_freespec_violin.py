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

class ViolinFreespecResult(EnterpriseWarpResult):
  def __init__(self, opts):
    super(ViolinFreespecResult, self).__init__(opts)
    self.inv_earth_orb_period = (365.25*24.*60.*60)**(-1)
    self.inv_mer_orb_period = (87.969*24.*60.*60)**(-1)
    self.inv_ven_orb_period = (224.7*24.*60.*60)**(-1)
    self.inv_mar_orb_period = (687.*24.*60.*60)**(-1)
    self.inv_jup_orb_period = (11.86 * 365.25 * 24.*60.*60)**(-1)
    self.inv_sat_orb_period = (29.457 * 365.25 * 24.*60.*60)**(-1)

  def main_pipeline(self, plot=True):
    for psr_dir in self.psr_dirs:

      self.psr_dir = psr_dir
      success = self._scan_psr_output()
      if not success:
        continue

      if not (self.opts.par):
        continue

      success = self.load_chains()
      if not success:
        continue

      self._get_par_mask()
      if plot:
        self._make_violin_plot()

  def make_figure(self, ax, label='Factorized posterior', facecolor='orange', 
                  edgecolor=None, alpha=0.3, linewidth=1, bw_method=None,
                  df = 2.2292808626385243e-09, yval='log10_rho'):
    freqs = np.arange(df, (np.sum(self.par_mask) + 0.1)*df, df)
    if yval=='log10_rho':
      fobj = ax.violinplot(self.chain_burn[:,self.par_mask], positions=freqs, \
                           widths=df, showextrema=False, bw_method=bw_method)
    elif yval=='log10_psd':
      fobj = ax.violinplot(2.*self.chain_burn[:,self.par_mask] - np.log10(df), \
                           positions=freqs, widths=df, showextrema=False, \
                           bw_method=bw_method)
    for pc in fobj['bodies']:
      pc.set_facecolor(facecolor)
      pc.set_edgecolor(edgecolor)
      pc.set_linewidth(linewidth)
      pc.set_alpha(alpha)
    return fobj

  def _make_violin_plot(self):
    fig, axes = plt.subplots()
    self.make_figure(axes)
    #chain_plot = self.chain_burn[:,self.par_mask]
    #rho = 10**chain_plot
    #df = 2.2292808626385243e-09 # x_1
    #freqs = np.arange(df,30.1*df,df)

    #fig, (ax1) = plt.subplots(nrows=1, ncols=1)
    #ax1.violinplot(rho, positions=freqs, widths=df, showextrema=False)
    #ax1.set_yscale('log')
    #ax1.set_xscale('log')
    #plt.xlabel('Frequency, Hz')
    #plt.ylabel('$\\rho\\mathrm{, s}$')
    #plt.tight_layout()
    #plt.savefig(self.outdir + 'freespec_1.png')
    #plt.close()

    #fig, (ax1) = plt.subplots(nrows=1, ncols=1)
    #ax1.violinplot(chain_plot, positions=freqs, widths=df, showextrema=False)
    #ax1.axvline(x=self.inv_earth_orb_period, label='$\mathrm{yr}^{-1}$')
    axes.set_xscale('log')
    plt.xlabel('Frequency, Hz')
    plt.ylabel('$\\log_{10}\\rho\\mathrm{, }\\log_{10}\\mathrm{s}$')
    plt.tight_layout()
    plt.savefig(self.outdir + 'freespec_2.png')
    plt.close()

def main():
  """
  The pipeline script
  """
  opts = parse_commandline()

  result_obj = ViolinFreespecResult(opts)
  result_obj.main_pipeline()#, inc_psrs=inc_psrs)

if __name__=='__main__':
  main()

