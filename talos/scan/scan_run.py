from tqdm import tqdm

from datetime import datetime

from ..utils.results import result_todf, peak_epochs_todf
from .scan_round import scan_round
from .scan_finish import scan_finish


def scan_run(self):

    '''The high-level management of the scan procedures
    onwards from preparation. Manages round_run()'''
    
    if self.max_iteration_start_time != None:
        stoptime=datetime.strptime(self.max_iteration_start_time,"%Y-%m-%d %H:%M")

    # main loop for the experiment
    # NOTE: the progress bar is also updated on line 73
    self.pbar = tqdm(total=len(self.param_log),
                     disable=self.disable_progress_bar)
    while len(self.param_log) != 0:
        self = scan_round(self)
        self.pbar.update(1)
        if self.max_iteration_start_time != None and datetime.now() > stoptime:
            print("Time limit reached, experiment finished")
            break
    self.pbar.close()

    # save the results
    self = result_todf(self)
    self.peak_epochs_df = peak_epochs_todf(self)

    self = scan_finish(self)
