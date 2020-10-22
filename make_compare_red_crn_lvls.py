import os
import json
import glob
import numpy as np
from matplotlib import pyplot as plt

from enterprise_warp.results import suitable_estimator

crn_lvl_dir = '/fred/oz002/bgoncharov/correlated_noise_pta_2020_out/dr2_timing_20200607/20200908_20200917_cpl_vargam_x1psr/noisefiles/'
pe_lvl_dir = '/fred/oz002/bgoncharov/correlated_noise_pta_2020_out/dr2_timing_20200607/20200908_v1/noisefiles/'

with open(crn_lvl_dir+'_credlvl.json', 'r') as fin:
  crn_lvl = json.load(fin)

pe_lvl = {}
lvlfiles = sorted(glob.glob(pe_lvl_dir + '*_credlvl.json'))
for lf in lvlfiles:
  psr_name = os.path.basename(lf)[:-13].split('_')[1]
  pe_lvl[psr_name] = {}
  with open(lf, 'r') as fin:
    pe_lvl[psr_name] = json.load(fin)

'red_noise'
'system_noise_XXX_SYSTEM'
'band_noise_by_B_1020CM'
'band_noise_XXX_20CM'

keystr_red = ['red_noise','system_noise','band_noise']

yy_1 = list()
yy_met_1 = list()
yy_err_1 = list()
yy_2 = list()
yy_met_2 = list()
yy_err_2 = list()
xx = list()
xx_lbl_1 = list()
xx_lbl_2 = list()
count = 0
for psr in pe_lvl.keys():
  for param in pe_lvl[psr].keys():
    #if any(map(param.__contains__, keystr_red)) and 'log10_A' in param:
    if 'red_noise' in param and 'log10_A' in param:
    #if ('band_noise' in param or 'system_noise' in param) and 'log10_A' in param:
    #if 'chromatic_gp' in param and 'log10_A' in param:

      (center_val, method) = suitable_estimator(pe_lvl[psr][param])

      yy_1.append(center_val)
      yy_met_1.append(method)
      yy_err_1.append([pe_lvl[psr][param]['84'] - center_val, \
                       center_val - pe_lvl[psr][param]['16']])
      xx.append(count)
      xx_lbl_1.append(param)
      xx_lbl_2.append(pe_lvl[psr][param.replace('log10_A','gamma')]['maximum'])
      count += 1

      # Common red noise case
      if param in crn_lvl.keys():
        (center_val, method) = suitable_estimator(crn_lvl[param])
        yy_2.append(center_val)
        yy_met_2.append(method)
        yy_err_2.append([crn_lvl[param]['84'] - center_val, \
                         center_val - crn_lvl[param]['16']])
      else:
        yy_2.append(np.nan)
        yy_met_2.append(np.nan)
        yy_err_2.append([np.nan,np.nan])
        print('Warning: parameter ', param, 'not in CRN!')

xx_lbl_2 = [str("{:.1f}".format(xx2)) for xx2 in xx_lbl_2]

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(111)
plt.errorbar(np.array(xx)-0.05, yy_1, yerr=np.array(yy_err_1).T, linestyle='', fmt='.', color='red', label='No CRN')
plt.errorbar(np.array(xx)+0.05, yy_2, yerr=np.array(yy_err_2).T, linestyle='', fmt='.', color='blue', label='CNR')
plt.xticks(xx, xx_lbl_1, rotation='vertical')
#ax1.set_xticks(xx)
#ax1.set_xticklabels(xx_lbl_1)
#plt.setp( ax1.xaxis.get_majorticklabels(), rotation=70 )
plt.xlabel('Red noise parameter')
plt.ylabel('$\log_{10}A$')
plt.legend()
ax2 = ax1.twiny()
ax2.set_xticks(xx)
ax2.set_xticklabels(xx_lbl_2)
ax2.set_xlabel('$\gamma$')
ax1.set_xlim([np.min(xx)-0.5, np.max(xx)+0.5])
ax2.set_xlim([np.min(xx)-0.5, np.max(xx)+0.5])
plt.tight_layout()
plt.savefig(crn_lvl_dir + 'compare_red_crn_levels.png')
plt.close()

import ipdb; ipdb.set_trace()
