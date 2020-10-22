import scipy
import warnings
import numpy as np
import enterprise.constants as const
from enterprise.signals import signal_base
from enterprise_extensions import chromatic
import enterprise.signals.parameter as parameter
import enterprise.signals.gp_signals as gp_signals
import enterprise.signals.deterministic_signals as deterministic_signals
import enterprise.signals.selections as selections
import enterprise.signals.utils as utils

from enterprise_warp.enterprise_models import StandardModels
from enterprise_warp.enterprise_models import selection_factory

class PPTADR2Models(StandardModels):
  """
  Please follow this example to add your own models for enterprise_warp.
  """
  def __init__(self,psr=None,params=None):
    super(PPTADR2Models, self).__init__(psr=psr,params=params)
    self.priors.update({
      "fd_sys_slope_range": 1e-7,
      "event_j0437_t0": [57050., 57150.],
      "event_j1643_t0": [57050., 57150.],
      "event_j1713_1_t0": [54500., 54900.],
      "event_j1713_2_t0": [57500., 57520.],
      "event_j2145_t0": [56100., 56500.],
      "event_j1603_t0": [53710., 54070.]
    })

  def dm_annual(self, option="default"):
    if option=="default": idx = 2
    return dm_annual_signal(idx=idx)
  
  def fd_sys_g(self,option=[]):
  
    idx = 1 # fitting linear trend 
    for ii, fd_sys_term in enumerate(option):
      name = 'fd' + str(idx) + '_sys_' + fd_sys_term
      slope = parameter.Uniform(-self.params.fd_sys_slope_range,\
                                 self.params.fd_sys_slope_range)
      wf = fd_system(slope = slope, idx_fd = idx)
  
      selection_function_name = 'fd_system_selection_' + \
                                 str(self.sys_noise_count)
      setattr(self, selection_function_name,
              selection_factory(selection_function_name))
      if fd_sys_term == "WBCORR_10CM_512":
        self.psr.sys_flags.append('beconfig')
        self.psr.sys_flagvals.append('wbb256_512_256_3p_b')
      else:
        self.psr.sys_flags.append('group')
        self.psr.sys_flagvals.append(fd_sys_term)
  
      fd_sys_term = deterministic_signals.Deterministic( wf, name=name,
                    selection=selections.Selection(\
                    self.__dict__[selection_function_name]) )
  
      if ii == 0:
        fd_sys = fd_sys_term
      elif ii > 0:
        fd_sys += fd_sys_term
  
      self.sys_noise_count += 1
  
    return fd_sys

  def j0437_event(self, option="exp_dip"):
    return dm_exponential_dip(self.params.event_j0437_t0[0],
                              self.params.event_j0437_t0[1],
                              idx="vary", tau_min_10_pow=5, tau_max_10_pow=100)
  
  def j1713_event_1(self, option="exp_dip"):
    return dm_exponential_dip(self.params.event_j1713_1_t0[0],
                              self.params.event_j1713_1_t0[1], idx=2,
                              tau_min_10_pow=5, tau_max_10_pow=1000,
                              name='dmexp_1')
  
  def j1713_event_2(self, option="exp_dip"):
    return dm_exponential_dip(self.params.event_j1713_2_t0[0],
                              self.params.event_j1713_2_t0[1], idx="vary",
                              tau_min_10_pow=5, tau_max_10_pow=100,
                              name='dmexp_2')
  
  def j1643_event(self, option="exp_dip"):
    return dm_exponential_dip(self.params.event_j1643_t0[0],
                              self.params.event_j1643_t0[1],
                              idx="vary", tau_min_10_pow=5, tau_max_10_pow=1000)
  
  def j2145_event(self, option="exp_dip"):
    return dm_exponential_dip(self.params.event_j2145_t0[0],
                              self.params.event_j2145_t0[1],
                              idx="vary", tau_min_10_pow=5, tau_max_10_pow=1000)
  
  def j1603_event(self, option="gaussian_bump"):
    return dm_gaussian_bump(self.params.event_j1603_t0[0],
                            self.params.event_j1603_t0[1], idx=2)

  def paired_ppta_band_noise(self, option=[]):
    """
    Including band noise terms for paired values of the PPTA "-B" flag:
    joint 1020-cm and 4050-cm red processes.
    """
    for ii, paired_band_term in enumerate(option):

      log10_A = parameter.Uniform(self.params.syn_lgA[0],self.params.syn_lgA[1])
      gamma = parameter.Uniform(self.params.syn_gamma[0],\
                                self.params.syn_gamma[1])
      pl = utils.powerlaw(log10_A=log10_A, gamma=gamma, \
                          components=self.params.red_general_nfouriercomp)
  
      setattr(self, paired_band_term, globals()[paired_band_term])
      nfreqs = self.determine_nfreqs(sel_func_name=paired_band_term)
  
      pbn_term = gp_signals.FourierBasisGP(spectrum=pl, Tspan=self.params.Tspan,
                                        name='band_noise_' + paired_band_term,
                                        selection=selections.Selection( \
                                        self.__dict__[paired_band_term] ),
                                        components=nfreqs)
      if ii == 0:
        pbn = pbn_term
      elif ii > 0:
        pbn += pbn_term

    return pbn

  def bayes_ephem(self,option="default"):
    """
    Deterministic signal from errors in Solar System ephemerides.
    """
    ekw = {}
    ekw['model'] = "orbel-v2"
    if isinstance(option, dict):
      # Converting parameters to bool masks
      for bekey, beval in option.items():
        if isinstance(beval, list):
          option[bekey] = np.array(beval, dtype=bool)
        else:
          option[bekey] = bool(beval)
      options = '_'.join(option.keys())
    else:
      options = option

    if "framedr" in option:
      ekw['frame_drift_rate'] = parameter.Uniform(-1e-10, 1e-10)\
                                                 ('frame_drift_rate')

    if "mer_m" in option or "inner" in option:
      ekw['d_mercury_mass'] = parameter.Normal(0, 1.66-10)('d_mer_mass')
    if "mer_el" in option:
      if isinstance(option, dict):
        ekw['mer_orb_elements'] = UniformMask(-0.5, 0.5, option['mer_el'])\
                                             ('mer_oe')
      else:
        ekw['mer_orb_elements'] = parameter.Uniform(-0.5, 0.5, size=6)('mer_oe')

    if "ven_m" in options or "inner" in options:
      ekw['d_venus_mass'] = parameter.Normal(0, 2.45e-9)('d_ven_mass')
    if "ven_el" in options:
      if isinstance(option, dict):
        ekw['ven_orb_elements'] = UniformMask(-1., 1., option['ven_el'])('ven_oe')
      else:
        ekw['ven_orb_elements'] = parameter.Uniform(-1., 1., size=6)('ven_oe')

    if "mar_m" in option or "inner" in option:
      ekw['d_mars_mass'] = parameter.Normal(0, 3.23e-10)('d_mar_mass')
    if "mar_el" in option:
      if isinstance(option, dict):
        ekw['mar_orb_elements'] = UniformMask(-5., 5., option['mar_el'])('mar_oe')
      else:
        ekw['mar_orb_elements'] = parameter.Uniform(-5., 5., size=6)('mar_oe')

    if "jup_m" in option or "outer" in option or "default" in option:
      ekw['d_jupiter_mass'] = parameter.Normal(0, 1.54976690e-11)\
                                              ('d_jup_mass')
    if "jup_el" in option or "outer" in option or "default" in option:
      if isinstance(option, dict):
        ekw['jup_orb_elements'] = UniformMask(-0.05, 0.05, option['jup_el'])\
                                             ('jup_oe')
      else:
        ekw['jup_orb_elements'] = parameter.Uniform(-0.05, 0.05, size=6)\
                                                   ('jup_oe')

    if "sat_m" in option or "outer" in option or "default" in option:
      ekw['d_saturn_mass'] = parameter.Normal(0, 8.17306184e-12)('d_sat_mass')
    if "sat_el" in option or "outer" in option:
      if isinstance(option, dict):
        ekw['sat_orb_elements'] = UniformMask(-5., 5., option['sat_el'])('sat_oe')
      else:
        ekw['sat_orb_elements'] = parameter.Uniform(-5., 5., size=6)('sat_oe')

    if "ura_m" in option or "outer" in option or "default" in option:
      ekw['d_uranus_mass'] = parameter.Normal(0, 5.71923361e-11)('d_ura_mass')
    if "ura_el" in option:
      if isinstance(option, dict):
        ekw['ura_orb_elements'] = UniformMask(-.5, .5, option['ura_el'])('ura_oe')
      else:
        ekw['ura_orb_elements'] = parameter.Uniform(-0.5, 0.5, size=6)('ura_oe')

    if "nep_m" in option or "outer" in option or "default" in option:
      ekw['d_neptune_mass'] = parameter.Normal(0, 7.96103855e-11)\
                                              ('d_nep_mass')
    if "nep_el" in option:
      if isinstance(option, dict):
        ekw['nep_orb_elements'] = UniformMask(-.5, .5, option['nep_el'])('nep_oe')
      else:
        ekw['nep_orb_elements'] = parameter.Uniform(-0.5, 0.5, size=6)('nep_oe')

    eph = deterministic_signals.PhysicalEphemerisSignal(**ekw)
    return eph

# PPTA DR2 signal models

@signal_base.function
def fd_system(freqs, slope = 1e-7, idx_fd = 1):
    freq_median = freqs - np.median(freqs)
    return np.sign(freq_median)**(idx_fd + 1) * slope * freq_median**idx_fd

@signal_base.function
def chrom_gaussian_bump(toas, freqs, log10_Amp=-2.5, sign_param=1.0,
                    t0=53890, sigma=81, idx=2):
    """
    Chromatic time-domain Gaussian delay term in TOAs.
    Example: J1603-7202 in Lentati et al, MNRAS 458, 2016.
    """
    t0 *= const.day
    sigma *= const.day
    wf = 10**log10_Amp * np.exp(-(toas - t0)**2/2/sigma**2)
    return np.sign(sign_param) * wf * (1400 / freqs) ** idx

# PPTA DR2 signal wrappers

def dm_exponential_dip(tmin, tmax, idx=2, sign='negative', name='dmexp',
                       lgA_min=-10., lgA_max=-5., sign_min=-1., sign_max=1.,
                       tau_min_10_pow=5, tau_max_10_pow=100):
    t0_dmexp = parameter.Uniform(tmin,tmax)
    log10_Amp_dmexp = parameter.Uniform(lgA_min, lgA_max)
    log10_tau_dmexp = parameter.Uniform(np.log10(tau_min_10_pow),
                                        np.log10(tau_max_10_pow))
    if idx=='vary':
        idx = parameter.Uniform(-7, 7)
    if sign == 'vary':
        sign_param = parameter.Uniform(sign_min, sign_max)
    elif sign == 'positive':
        sign_param = 1.0
    else:
        sign_param = -1.0
    wf = chromatic.chrom_exp_decay(log10_Amp=log10_Amp_dmexp,
                                t0=t0_dmexp, log10_tau=log10_tau_dmexp,
                                sign_param=sign_param, idx=idx)
    dmexp = deterministic_signals.Deterministic(wf, name=name)

    return dmexp

def dm_gaussian_bump(tmin, tmax, idx=2, sigma_min=20, sigma_max=140,
    log10_A_low=-6, log10_A_high=-5, name='dm_bump'):
    """
    For the extreme scattering event in J1603-7202.
    """
    sign_param = 1.0
    t0_dm_bump = parameter.Uniform(tmin,tmax)
    sigma_dm_bump = parameter.Uniform(sigma_min,sigma_max)
    log10_Amp_dm_bump = parameter.Uniform(log10_A_low, log10_A_high)
    if idx == 'vary':
        idx = parameter.Uniform(0, 6)
    wf = chrom_gaussian_bump(log10_Amp=log10_Amp_dm_bump,
                         t0=t0_dm_bump, sigma=sigma_dm_bump,
                         sign_param=sign_param, idx=idx)
    dm_bump = deterministic_signals.Deterministic(wf, name=name)

    return dm_bump

def dm_annual_signal(idx=2, name='dm_s1yr'):
    """
    Returns chromatic annual signal (i.e. TOA advance)
    """
    log10_Amp_dm1yr = parameter.Uniform(-10., -5.)
    phase_dm1yr = parameter.Uniform(0, 2*np.pi)

    wf = chromatic.chrom_yearly_sinusoid(log10_Amp=log10_Amp_dm1yr,
                                      phase=phase_dm1yr, idx=idx)
    dm1yr = deterministic_signals.Deterministic(wf, name=name)

    return dm1yr

# PPTA DR2 selections

def by_B_4050CM(flags):
    """Selection function to split by PPTA frequency band under -B flag"""
    sel_40b = np.char.lower(flags['B']) == '40cm'
    sel_50b = np.char.lower(flags['B']) == '50cm'
    return {'4050CM': sel_40b + sel_50b}

def by_B_1020CM(flags):
    """Selection function to split by PPTA frequency band under -B flag"""
    sel_10b = np.char.lower(flags['B']) == '10cm'
    sel_20b = np.char.lower(flags['B']) == '20cm'
    return {'1020CM': sel_10b + sel_20b}

# Custom priors

#def UniformMaskPrior(value, pmin, pmax, mask):
#    """Prior function for Uniform parameters."""
#    print('Warning! Does not work: pmin and pmax are passed OK as both \
#           args and kwargs, but mask is passed as neither.')
#    return np.ma.array(
#           scipy.stats.uniform.pdf(value, pmin, pmax - pmin),
#           mask=~mask, fill_value=1.,).filled()

def deltafunc(val):
    return ~bool(val)

def UniformMaskSampler(pmin, pmax, mask, size=None):
    """Sampling function for Uniform parameters."""
    return np.ma.array(
           scipy.stats.uniform.rvs(pmin, pmax - pmin, size=len(mask)),
           mask=~mask, fill_value=0.,).filled()

def UniformMask(pmin, pmax, mask):
    """
    Similar to enterprise.signals.parameter.Uniform, but with an option to
    make some of the iterable sub-parameters constant, using a mask.
    """
    #warnings.warn("Make sure that parameters are fixed at values, which are \
    #               within the uniform prior range: otherwise \
    #               parameter.UniformPrior will yield incorrect probability.")

    def UniformMaskPrior(value, pmin, pmax, mask=mask):
        """Prior function for Uniform parameters."""
        output = scipy.stats.uniform.pdf(value, pmin, pmax - pmin)
        fill_val_delta = (~value[~mask].astype(bool)).astype(float)
        output[~mask] = fill_val_delta
        return output
    #return np.ma.array(
    #       scipy.stats.uniform.pdf(value, pmin, pmax - pmin),
    #       mask=~mask, fill_value=fill_val_delta,).filled()

    class UniformMask(parameter.Parameter):
        _size = len(mask) #size
        _prior = parameter.Function(UniformMaskPrior, pmin=pmin, 
                                    pmax=pmax, mask=mask)
        _sampler = staticmethod(UniformMaskSampler)
        _typename = parameter._argrepr("UniformMask", pmin=pmin, pmax=pmax,
                                       mask=mask)

    return UniformMask
