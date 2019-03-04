from collections import OrderedDict

from .scan_prepare import scan_prepare
from .scan_run import scan_run


class Scan:
    """Hyperparamater scanning and optimization

    USE: ta.Scan(x=x, y=y, params=params_dict, model=model)

    Takes in a Keras model, and a dictionary with the parameter
    boundaries for the experiment.

        p = {
            'epochs' : [50, 100, 200],
            'activation' : ['relu'],
            'dropout': (0, 0.1, 5)
        }

    Accepted input formats are [1] single value in a list, [0.1, 0.2]
    multiple values in a list, and (0, 0.1, 5) a range of 5 values
    from 0 to 0.1.

    Here is an example of the input model:

    def model():

        # any Keras model

        return out, model


    You must replace the parameters in the model with references to
    the dictionary, for example:

    model.fit(epochs=params['epochs'])

    To learn more, start from the examples and documentation
    available here: https://github.com/autonomio/talos


    PARAMETERS
    ----------
    x : ndarray
        1d or 2d array consisting of the training data. `x` should have the
        shape (m, n), where m is the number of training examples and n is the
        number of features. Extra dimensions can be added to account for the
        channels entry in convolutional neural networks.
    y : ndarray
        The labels corresponding to the training data. `y` should have the
        shape (m, c) where c is the number of classes. A binary classification
        problem will have c=1.
    params : python dictionary
        Lists all permutations of hyperparameters, a subset of which will be
        selected at random for training and evaluation.
    model : keras model
        Any Keras model with relevant declrations like params['first_neuron']
    dataset_name : str
        References the name of the experiment. The dataset_name and
        experiment_no will be concatenated to produce the file name for the
        results saved in the local directory.
    experiment_no : str
        Indexes the user's choice of experiment number.
    x_val : ndarray
        User specified cross-validation data. (Default is None).
    y_val : ndarray
        User specified cross-validation labels. (Default is None).
    val_split : float, optional
        The proportion of the input `x` which is set aside as the
        validation data. (Default is 0.3).
    shuffle : bool, optional
        If True, shuffle the data in x and y before splitting into the train
        and cross-validation datasets. (Default is True).
    random_method : uniform, stratified, lhs, lhs_sudoku
        Determinines the way in which the grid_downsample is applied. The
        default setting is 'uniform'.
    seed : int
        Sets numpy random seed.
    search_method : {None, 'random', 'linear', 'reverse'}
        Determines the random sampling of the dictionary. `random` picks one
        hyperparameter point at random and removes it from the list, then
        samples again. `linear` starts from the start of the grid and moves
        forward, and `reverse` starts at the end of the grid and moves
        backwards.
    max_iteration_start_time : None or str
        Allows setting a time when experiment will be completed. Use the format
        "%Y-%m-%d %H:%M" here.
    permutation_filter : lambda function
        Use it to filter permutations based on previous knowledge.
        USE: permutation_filter=lambda p: p['batch_size'] < 150
        This example removes any permutation where batch_size is below 150
    reduction_method : {None, 'correlation'}
        Method for honing in on the optimal hyperparameter subspace. (Default
        is None).
    reduction_interval : int
        The number of reduction method rounds that will be performed. (Default
        is None).
    reduction_window : int
        The number of rounds of the reduction method before observing the
        results. (Default is None).
    grid_downsample : int
        The fraction of `params` that will be tested (Default is None).
    round_limit : int
        Limits the number of rounds (permutations) in the experiment.
    reduction_metric : {'val_acc'}
        Metric used to tune the reductions.
    last_epoch_value : bool
        Set to True if the last epoch metric values are logged as opposed
        to the default which is peak epoch values for each round.
    disable_progress_bar : bool
        Disable TQDM live progress bar.
    print_params : bool
        Print params for each round on screen (useful when using TrainingLog
        callback for visualization)
    debug : bool
        Implements debugging feedback. (Default is False).

    """

    # TODO: refactor this so that we don't initialize global variables
    global self

    def __init__(self, x, y, params, model,
                 dataset_name=None,
                 experiment_no=None,
                 experiment_name=None,
                 x_val=None,
                 y_val=None,
                 val_split=.3,
                 shuffle=True,
                 round_limit=None,
                 time_limit=None,
                 grid_downsample=1.0,
                 random_method='uniform_mersenne',
                 seed=None,
                 search_method='random',
                 permutation_filter=None,
                 reduction_method=None,
                 reduction_interval=50,
                 reduction_window=20,
                 reduction_threshold=0.2,
                 reduction_metric='val_acc',
                 reduce_loss=False,
                 last_epoch_value=False,
                 clear_tf_session=True,
                 disable_progress_bar=False,
                 print_params=False,
                 debug=False):

        # NOTE: these need to be follow the order from __init__
        # and all paramaters needs to be included here and only here.

        self.x = x
        self.y = y
        self.params = OrderedDict(params)
        self.model = model
        self.dataset_name = dataset_name
        self.experiment_no = experiment_no
        self.experiment_name = experiment_name
        self.x_val = x_val
        self.y_val = y_val
        self.val_split = val_split
        self.shuffle = shuffle
        self.random_method = random_method
        self.search_method = search_method
        self.round_limit = round_limit
        self.time_limit = time_limit
        self.permutation_filter = permutation_filter
        self.reduction_method = reduction_method
        self.reduction_interval = reduction_interval
        self.reduction_window = reduction_window
        self.grid_downsample = grid_downsample
        self.reduction_threshold = reduction_threshold
        self.reduction_metric = reduction_metric
        self.reduce_loss = reduce_loss
        self.debug = debug
        self.seed = seed
        self.clear_tf_session = clear_tf_session
        self.disable_progress_bar = disable_progress_bar
        self.last_epoch_value = last_epoch_value
        self.print_params = print_params
        # input parameters section ends

        self._null = self.runtime()

    def runtime(self):

        self = scan_prepare(self)
        self = scan_run(self)
