from matplotlib import pyplot as plt
import numpy as np

def hd_orf(eta):
  """ Angle eta in rad """
  comb = (1. - np.cos(eta))/2.
  return 0.5 - 0.25*comb + 1.5*comb*np.log(comb)

def dipole(eta):
  return np.cos(eta)

def monopole(eta):
  return np.repeat(1.0, len(eta))

output_directory = '/home/bgonchar/correlated_noise_pta_2020/publication_figures/'

eta = np.linspace(0,np.pi,100)

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif",
  #"font.serif": ["Palatino"],
})
font = {'family' : 'serif',
        'size'   : 17}

fig = plt.figure()
axes = fig.add_subplot(111)

plt.plot(eta, monopole(eta), c="#590a30", linewidth=3, label='Monopole')
plt.plot(eta, dipole(eta), c="#90aa3c", linewidth=3, label='Dipole')
plt.plot(eta, hd_orf(eta), c="#ef6125", linewidth=3, label='Hellings-Downs')

axes.set_xlabel('$\\mathrm{Angle~between~Earth-pulsar~baselines,~} \\zeta \\mathrm{~[deg]}$', fontdict=font)
axes.set_ylabel('$\\mathrm{Spatial~correlation,~}\\Gamma(\\zeta)$', fontdict=font)

plt.xlim([0, np.pi])
axes.grid(False)
axes.tick_params(axis='x', labelsize = font['size'])
axes.tick_params(axis='y', labelsize = font['size'])
plt.legend(prop=font)
plt.tight_layout()
plt.savefig(output_directory + 'illustration_orf.pdf')
plt.close()
