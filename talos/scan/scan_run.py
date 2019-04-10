def scan_run(self):

    '''The high-level management of the scan procedures
    onwards from preparation. Manages round_run()'''

    from tqdm import tqdm

    from .scan_prepare import scan_prepare
    self = scan_prepare(self)

    # initiate the progress bar
    self.pbar = tqdm(total=len(self.param_object.param_index),
                     disable=self.disable_progress_bar)

    # the main cycle of the experiment
    while True:

        # get the parameters
        self.round_params = self.param_object.round_parameters()

        # break when there is no more permutations left
        if self.round_params is False:
            break
        # otherwise proceed with next permutation
        from .scan_round import scan_round
        self = scan_round(self)
        self.pbar.update(1)

    # close progress bar before finishing
    self.pbar.close()

    # finish
    from ..logging.logging_finish import logging_finish
    self = logging_finish(self)

    from .scan_finish import scan_finish
    self = scan_finish(self)
