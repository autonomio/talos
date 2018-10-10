from tqdm import tqdm

from ..utils.results import result_todf, peak_epochs_todf
from .scan_round import scan_round
from .scan_finish import scan_finish

def scan_run(self):

    '''The high-level management of the scan procedures
    onwards from preparation. Manages round_run()'''

    # enforce round_limit
    self.param_grid = self.param_grid[:self.round_limit]

    # main loop for the experiment
    # NOTE: the progress bar is also updated on line 73
    self.pbar = tqdm(total=len(self.param_log),
                     disable=self.disable_progress_bar)
    while len(self.param_log) != 0:
        self = scan_round(self)
        self.pbar.update(1)
    self.pbar.close()

    # save the results
    self = result_todf(self)
    self.peak_epochs_df = peak_epochs_todf(self)

    self = scan_finish(self)

    print('Scan Finished!')
