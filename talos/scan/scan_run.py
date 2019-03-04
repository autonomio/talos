from tqdm import tqdm
from datetime import datetime

from ..utils.results import result_todf, peak_epochs_todf
from .scan_round import scan_round
from .scan_finish import scan_finish


def scan_run(self):

    '''The high-level management of the scan procedures
    onwards from preparation. Manages round_run()'''

    # initiate the progress bar
    self.pbar = tqdm(total=len(self.param_log),
                     disable=self.disable_progress_bar)

    # start the main loop of the program
    while len(self.param_log) != 0:
        self = scan_round(self)
        self.pbar.update(1)
        if self.time_limit is not None:
            if datetime.now() > self._stoptime:
                print("Time limit reached, experiment finished")
                break
    self.pbar.close()

    # save the results
    self = result_todf(self)
    self.peak_epochs_df = peak_epochs_todf(self)

    self = scan_finish(self)
