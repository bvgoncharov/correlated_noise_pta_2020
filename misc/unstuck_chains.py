import os
import numpy as np
from datetime import datetime
from shutil import copyfile
from enterprise_warp.results import EnterpriseWarpResult
from enterprise_warp.results import parse_commandline

class ChainsStatus(EnterpriseWarpResult):
  def __init__(self, opts):
    super(ChainsStatus, self).__init__(opts)

  def main_pipeline(self):
    for psr_dir in self.psr_dirs:
      self.psr_dir = psr_dir

      success = self._scan_psr_output()
      if not success:
        continue

      success = self.load_chains()
      if not success:
        continue

      self.compare_chain_iterations()
      self.save_new_chain()

  def save_new_chain(self):
    fname_split = os.path.splitext(self.chain_file)
    dtime = datetime.now().strftime("%d%m%Y%H%M%S")
    copyfile(self.chain_file, fname_split[0] + '_' + dtime + fname_split[1])
    np.savetxt(self.chain_file, self.chain[:self.keep_rows,:])
    print('New chain saved: ', self.chain_file)

  def compare_chain_iterations(self):
    self.min_required_move = 2
    #self.chain_moved = [True]
    self.stuck_count = 0
    self.moved_count = 0
    for ii, row in enumerate(reversed(self.chain[1:,:])):
      if any(row[:-5] != self.chain[1:,:-5][-ii-2]): # -5: 4 last columns+nmodel
        self.moved_count += 1
        if self.moved_count >= self.min_required_move:
          break
      else:
        self.stuck_count += 1
        if self.moved_count < self.min_required_move:
          self.stuck_count += self.moved_count
          self.moved_count = 0
    self.keep_rows = self.chain.shape[0] - self.stuck_count - 1
    print(self.keep_rows/self.chain.shape[0], ' of the chain will remain')
      #self.chain_moved.append( any(row != self.chain[1:,:][-ii-2]) )
    #self.chain_moved = np.array(self.chain_moved)
    

opts = parse_commandline()

result_obj = ChainsStatus(opts)
result_obj.main_pipeline()

import ipdb; ipdb.set_trace()
