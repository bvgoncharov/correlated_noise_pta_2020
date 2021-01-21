from matplotlib import pyplot as plt
import numpy as np
#from astropy.stats import LombScargle
from astropy.timeseries import LombScargle

from enterprise.pulsar import Tempo2Pulsar
import enterprise.constants as const
import libstempo.toasim as LT

from enterprise_warp import enterprise_warp
from enterprise.signals import utils

import ppta_dr2_models

#import sys
#sys.path.insert(0, "/home/bgonchar/run_etps/")
#import LT_custom

def calculate_psd_matlab(psr,datadir):
  ''' Calculate PSD by running MATLAB script separately '''
  [tosave_toas,tosave_res,tosave_toaerr] = preprocess_toas_for_psd( \
      psr.toas(),psr.residuals(),psr.toaerrs,precision='float64')

  pulsar_data = np.concatenate((tosave_toas,tosave_res,tosave_toaerr),axis=1)
  np.savetxt(datadir+'/tosave_data.txt',pulsar_data,delimiter=';')
  #os.system("python run_plomb.py")
  command = "matlab -nodesktop -nodisplay -r \"run_plomb(\'"+datadir+"\');exit;\""
  os.system(command)
  ml_psd_ff = np.loadtxt(datadir+'/toload_data.txt',delimiter=';')

  return ml_psd_ff

def determine_freqs(psrs, nfreqs=90):
  tmin = 1e40
  tmax = 0
  for psr in psrs:
    if np.min(psr.toas) < tmin: tmin = np.min(psr.toas)
    if np.max(psr.toas) > tmax: tmax = np.max(psr.toas)
  tobs = tmax - tmin
  return np.arange(1/tobs, (nfreqs+0.1)/tobs, 1/tobs)

output_paper = '/home/bgonchar/correlated_noise_pta_2020/publication_figures/'
output_temp = '/fred/oz002/bgoncharov/correlated_noise_pta_2020_out/bayesephem_psd_out/'

mode='libstempo'
if mode=='real':
  opts = enterprise_warp.parse_commandline()
  #opts.prfile = "/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_ms_common_pl_vargam_set_x_1_ephem_0_20200928.dat"
  opts.prfile = "/home/bgonchar/correlated_noise_pta_2020/params/ppta_dr2_ptmcmc_ms_common_pl_vargam_set_3_1_ephem_0_20200917.dat"
  
  custom = ppta_dr2_models.PPTADR2Models
  
  params = enterprise_warp.Params(opts.prfile,opts=opts,custom_models_obj=custom)
  pta = enterprise_warp.init_pta(params)

  toas = list()
  for psr in params.psrs:
    toas = np.append(toas, psr.toas)
    if 'pos_t' in locals():
      pos_t = np.append(pos_t, psr.pos_t, axis=0)
    else:
      pos_t = psr.pos_t
    if 'planetssb' in locals():
      planetssb = np.append(planetssb, psr.planetssb, axis=0)
    else:
      planetssb = psr.planetssb
  
elif mode=='libstempo':
  tt = np.arange(53041.16236381156,58229.94960008678,30.0) # Actual J1909 Tobs, 30-day cadence
  par = '/home/bgonchar/correlated_noise_pta_2020/data/dr2_nicer_nof2_20201028/J1909-3744.par'
  ltpsr = LT.fakepulsar(par,obstimes=tt,toaerr=0.01,flags='-nobs 10000000') # 0.01 us TOA error
  LT.add_equad(ltpsr,equad=1e-8) # 0.01 us EQUAD scatter
  #psd_noise = 2*(1e-8**2+1e-8**2)*(30*24*60*60) # 2 sigma^2 t_cadence
  psd_noise = 2*(1e-8**2)*(30*24*60*60) # 2 sigma^2 t_cadence
  psr = Tempo2Pulsar(ltpsr)
  toas = psr.toas
  dy = psr.toaerrs
  pos_t = psr.pos_t
  planetssb = psr.planetssb

defaults = {
    'frame_drift_rate': 0,
    'd_mercury_mass': 0,
    'd_venus_mass': 0,
    'd_mars_mass': 0,
    'd_jupiter_mass': 0,
    'd_saturn_mass': 0,
    'd_uranus_mass': 0,
    'd_neptune_mass': 0,
    'mer_orb_elements': 0,
    'ven_orb_elements': 0,
    'mar_orb_elements': 0,
    'jup_orb_elements': 0,
    'sat_orb_elements': 0,
    'ura_orb_elements': 0,
    'nep_orb_elements': 0,
}
demovals = {
    'frame_drift_rate': 1e-11,
    'd_mercury_mass': 1.66e-10,
    'd_venus_mass': 2.45e-9,
    'd_mars_mass': 3.23e-10,
    'd_jupiter_mass': 1.54976690e-11,
    'd_saturn_mass': 8.17306184e-12,
    'd_uranus_mass': 5.71923361e-11,
    'd_neptune_mass': 7.96103855e-11,
    'mer_orb_elements': np.repeat(10., 6),
    'ven_orb_elements': np.repeat(1., 6),
    'mar_orb_elements': np.repeat(5., 6),
    'jup_orb_elements': np.repeat(0.05, 6),
    'sat_orb_elements': np.repeat(0.1, 6),
    'ura_orb_elements': np.repeat(0.5, 6),
    'nep_orb_elements': np.repeat(0.5, 6),
}
times, mer_orbit, ven_orbit, \
    mar_orbit, jup_orbit, sat_orbit, \
    ura_orbit, nep_orbit = utils.get_planet_orbital_elements("orbel-v2")
demovals['times'], defaults['times'] = times, times
demovals['mer_orbit'], defaults['mer_orbit'] = mer_orbit, mer_orbit
demovals['ven_orbit'], defaults['ven_orbit'] = ven_orbit, ven_orbit
demovals['mar_orbit'], defaults['mar_orbit'] = mar_orbit, mar_orbit
demovals['jup_orbit'], defaults['jup_orbit'] = jup_orbit, jup_orbit
demovals['sat_orbit'], defaults['sat_orbit'] = sat_orbit, sat_orbit
demovals['ura_orbit'], defaults['ura_orbit'] = ura_orbit, ura_orbit
demovals['nep_orbit'], defaults['nep_orbit'] = nep_orbit, nep_orbit
inv_earth_orb_period = (365.25*24.*60.*60)**(-1)
inv_mer_orb_period = (87.969*24.*60.*60)**(-1)
inv_ven_orb_period = (224.7*24.*60.*60)**(-1)
inv_mar_orb_period = (687.*24.*60.*60)**(-1)
inv_jup_orb_period = (11.86 * 365.25 * 24.*60.*60)**(-1)
inv_sat_orb_period = (29.457 * 365.25 * 24.*60.*60)**(-1)
# Each config is one BayesEphem term
configs = list()
confnames = list()
for key, val in defaults.items():
  if 'elements' in key:
    for ii in range(6):
      config = defaults.copy()
      config[key] = np.zeros(6)
      config[key][ii] = demovals[key][ii]
      configs.append(config)
      confnames.append(key+'_'+str(ii))
  else:
    config = defaults.copy()
    config[key] = demovals[key]
    configs.append(config)
    confnames.append(key)

psd = dict()
freqs = determine_freqs([psr])
for ii, config in enumerate(configs):

  print('Config ', ii, '/',len(configs))

  res = utils.physical_ephem_delay(toas, planetssb, \
                                             pos_t, **config)
  #noise = np.random.normal(scale=1e-9,size=len(res))
  noise = psr.residuals
  res += noise
  power = LombScargle(toas-toas[0], res, normalization='psd').power(freqs)
  #ff, power = LombScargle(toas, res, normalization='psd').autopower()
  psd[ii] = power/np.sqrt(freqs)
  psd[ii] = psd[ii] * len(toas)*10 # MATLAB gives correct PSD, in AstroPy this weird factor needs to be applied - no idea
  #psd[ii] = power/np.sqrt(ff)
  noisepow = LombScargle(toas-toas[0], noise, normalization='psd').power(freqs)
  noisepsd = noisepow/np.sqrt(freqs)
  noisepsd = noisepsd * len(toas)*10 # MATLAB gives correct PSD, in AstroPy this weird factor needs to be applied - no idea
  #ffn, noisepow = LombScargle(toas, noise, normalization='psd').autopower()
  #noisepsd = noisepow/np.sqrt(ffn)

  fig, axes = plt.subplots()
  axes.axvline(x=inv_earth_orb_period, color='grey')
  axes.axvline(x=inv_sat_orb_period, color='grey')
  axes.axvline(x=inv_jup_orb_period, color='grey')
  axes.axvline(x=inv_mar_orb_period, color='grey')
  axes.axvline(x=inv_ven_orb_period, color='grey')
  axes.axvline(x=inv_mer_orb_period, color='grey', label='Inv. planet periods')
  plt.plot(freqs, psd[ii], marker='.', color='red', label=confnames[ii])
  plt.plot(freqs, noisepsd, marker='.', linestyle=':', color='grey', label='Noise only')
  axes.axhline(y=psd_noise, color='black', label='Noise PSD')
  #plt.xlim([np.min(freqs), np.max(freqs)])
  plt.xscale('log')
  plt.yscale('log')
  plt.xlabel('Frequency, Hz')
  plt.ylabel('$\\mathrm{PSD,~[s}^3\\mathrm{]}$')
  plt.legend()
  plt.tight_layout()
  plt.savefig(output_temp + 'config_'+str(ii)+'.png')
  plt.close()

import ipdb; ipdb.set_trace()
