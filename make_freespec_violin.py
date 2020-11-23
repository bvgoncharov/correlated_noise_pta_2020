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
    self.inv_mar_orb_period = (687.*24.*60.*60)**(-1)

  def main_pipeline(self):
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
      self._make_violin_plot()

  def _make_violin_plot(self):
    chain_plot = self.chain_burn[:,self.par_mask]
    rho = 10**chain_plot
    df = 2.2292808626385243e-09 # x_1
    freqs = np.arange(df,30.1*df,df)

    fig, (ax1) = plt.subplots(nrows=1, ncols=1)
    ax1.violinplot(rho, positions=freqs, widths=df, showextrema=False)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    plt.xlabel('Frequency, Hz')
    plt.ylabel('$\\rho\\mathrm{, s}$')
    plt.tight_layout()
    plt.savefig(self.outdir + 'freespec_1.png')
    plt.close()

    fig, (ax1) = plt.subplots(nrows=1, ncols=1)
    ax1.violinplot(chain_plot, positions=freqs, widths=df, showextrema=False)
    ax1.axvline(x=self.inv_earth_orb_period, label='$\mathrm{yr}^{-1}$')
    ax1.set_xscale('log')
    plt.xlabel('Frequency, Hz')
    plt.ylabel('$\\log_{10}\\rho\\mathrm{, }\\log_{10}\\mathrm{s}$')
    plt.tight_layout()
    plt.savefig(self.outdir + 'freespec_2.png')
    plt.close()

opts = parse_commandline()

result_obj = ViolinFreespecResult(opts)
#inc_psrs = ['20_J1909-3744','0_J0437-4715','24_J2145-0750','11_J1603-7202','5_J1024-0719','23_J2129-5721','22_J2124-3358','10_J1600-3053']
result_obj.main_pipeline()#, inc_psrs=inc_psrs)
