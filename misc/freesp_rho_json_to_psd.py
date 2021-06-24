import os
import json
import numpy as np

fname = '/fred/oz002/bgoncharov/correlated_noise_pta_2020_out/dr2_timing_20200607/20210120_cpl_freesp_30_nf_singpsr/noisefiles/20_J1909-3744_noise.json'
df = 2.2315554798611935e-09

with open(fname, 'r') as fin:
  all_json = json.load(fin)

rhos_ordered = []
count_rho = 0
exit = False
while not exit:
  key = 'gw_log10_rho_'+str(count_rho)
  if key in all_json.keys():
    rhos_ordered.append(all_json[key])
  else:
    break
  count_rho += 1

ff = np.arange(df, 30*df + df/100, df)
psd = 2.*np.array(rhos_ordered) - np.log10(df)

out = np.stack([ff,psd]).T

np.savetxt(os.path.dirname(fname)+'/1909-3744_f_log10psd.txt', out)

print('Completed')
