
def comment_out_toas(timfile_old, timfile_new, groups):
  tfn_lines = list()
  with open(timfile_old,"r") as tfo:
    for line in tfo:
      if 'MODE' in line or 'FORMAT' in line:
        tfn_lines.append(line)
        continue
      ls = line.split(' ')
      group_flagval = ls[ls.index('-group')+1]
      if group_flagval in groups:
        tfn_lines.append(line)
  with open(timfile_new,"w") as tfn:
    tfn.writelines(tfn_lines)


old_data_dir = '/home/bgonchar/correlated_noise_pta_2020/data/dr2_timing_20200607/'
new_data_dir = '/home/bgonchar/correlated_noise_pta_2020/data/dr2_boris_2_20201010/'
psr_tim = 'J0437-4715.tim'

comment_out_toas(old_data_dir+psr_tim, new_data_dir+psr_tim, ['PDFB1_10CM','PDFB_10CM','PDFB_40CM','CASPSR_40CM', 'CPSR2_50CM'])
