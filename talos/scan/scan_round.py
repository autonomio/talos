from time import strftime, time

from keras import backend as K

from ..parameters.round_params import round_params
from ..utils.results import create_header
from ..metrics.entropy import epoch_entropy
from ..model.ingest_model import ingest_model
from ..utils.results import run_round_results, save_result
from ..reducers.reduce_run import reduce_run
from ..utils.exceptions import TalosReturnError, TalosTypeError


def scan_round(self):

    '''The main operational function that manages the experiment
    on the level of execution of each round.'''

    # determine the parameters for the particular execution
    self.round_params = round_params(self)

    # print round params
    if self.print_params is True:
        print(self.round_params)

    # set start time
    round_start = strftime('%H%M%S')
    start = time()

    # fit the model
    try:
        _hr_out, self.keras_model = ingest_model(self)
    except TypeError as err:
        if err.args[0] == "unsupported operand type(s) for +: 'int' and 'numpy.str_'":
            raise TalosTypeError("Activation should be as object and not string in params")
        else:
            raise TalosReturnError("Make sure that input model returns 'out, model' where out is history object from model.fit()")

    # set end time and log
    round_end = strftime('%H%M%S')
    round_seconds = time() - start
    self.round_times.append([round_start, round_end, round_seconds])

    # create log and other stats
    try:
        self.epoch_entropy.append(epoch_entropy(_hr_out))
    except (TypeError, AttributeError):
        raise TalosReturnError("Make sure that input model returns in the order 'out, model'")

    if self.round_counter == 0:
        _for_header = create_header(self, _hr_out)
        self.result.append(_for_header)
        save_result(self)

    _hr_out = run_round_results(self, _hr_out)

    self.result.append(_hr_out)
    save_result(self)

    # apply reduction
    if self.reduction_method is not None:
        if (self.round_counter + 1) % self.reduction_interval == 0:
            len_before_reduce = len(self.param_log)
            self = reduce_run(self)
            total_reduced = len_before_reduce - len(self.param_log)
            # update the progress bar
            self.pbar.update(total_reduced)

    # save model and weights
    self.saved_models.append(self.keras_model.to_json())
    self.saved_weights.append(self.keras_model.get_weights())

    # clear tensorflow sessions (maybe)
    if self.clear_tf_session is True:
        K.clear_session()
    self.round_counter += 1

    return self
