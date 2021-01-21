
def comment_out_toas(timfile_old, timfile_new, n_subbands_max):
  tfn_lines = list()
  with open(timfile_old,"r") as tfo:
    prev_epoch_name = ''
    count_subbands = 0
    for line in tfo:
      if 'MODE' in line or 'FORMAT' in line:
        tfn_lines.append(line)
        continue
      ls = line.split(' ')
      this_epoch_name = ls[1]
      if this_epoch_name == prev_epoch_name:
        count_subbands += 1
        if count_subbands <= n_subbands_max:
          tfn_lines.append(line)
        else:
          continue
      else:
        tfn_lines.append(line)
        prev_epoch_name = this_epoch_name
        count_subbands = 1

  with open(timfile_new,"w") as tfn:
    tfn.writelines(tfn_lines)


old_data_dir = '/home/bgonchar/correlated_noise_pta_2020/data/dr2_timing_20200607/'
new_data_dir = '/home/bgonchar/correlated_noise_pta_2020/data/dr2_boris_4_20201010/'
psr_tim = 'J0437-4715.tim'

comment_out_toas(old_data_dir+psr_tim, new_data_dir+psr_tim, 3)
