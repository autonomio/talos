from .scan_prepare import scan_prepare
from .scan_run import scan_run


class Scan:
    """Suite of operations for training and evaluating Keras neural networks.

    Inputs train/dev data and a set of parameters as a dictionary. The name and
    experiment number must also be chosen since they define the output
    filenames. The model must also be specified of the form

        my_model(x_train, y_train, x_val, y_val, params),

    and the dictionary

        d = {
            'fcc_layer_1_N': [50, 100, 200],
            'fcc_layer_1_act': ['relu', 'tanh'],
            'fcc_layer_1_dropout': (0, 0.1, 5)    # 5 points between 0 and 0.1
        }

    The dictionary is parsed for every run and only one entry per parameter
    is fed into the neural network at a time.

    Important note: the user has two options when specifying input data.

    Option 1:
        Specify x, y and val_split. The training and validation data mixture
        (x, y) will be randomly split into the training and validation datasets
        as per the split specified in val_split.

    Option 2:
        Specify x, y and x_val, y_val. This would allow the user to specify
        their own validation datasets. Keras by default shuffles data during
        training, so the user need only be sure that the split specified is
        correct. This allows for not only reproducibility, but randomizing the
        data on the user's own terms. This is critical if the user wishes to
        augment their training data without augmenting their validation data
        (which is the only acceptable practice!).


    Parameters
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
    dataset_name : str
        References the name of the experiment. The dataset_name and
        experiment_no will be concatenated to produce the file name for the
        results saved in the local directory.
    experiment_no : str
        Indexes the user's choice of experiment number.
    model : keras_model
        A Keras style model which compiles and fits the data, and returns
        the history and compiled model.
    val_split : float, optional
        The proportion of the input `x` which is set aside as the
        cross-validation data. (Default is 0.3).
    shuffle : bool, optional
        If True, shuffle the data in x and y before splitting into the train
        and cross-validation datasets. (Default is True).
    random_method : uniform, stratified, lhs, lhs_sudoku
        Determinines the way in which the grid_downsample is applied. The
        default setting is 'uniform'.
    search_method : {None, 'random', 'linear', 'reverse'}
        Determines the random sampling of the dictionary. `random` picks one
        hyperparameter point at random and removes it from the list, then
        samples again. `linear` starts from the start of the grid and moves
        forward, and `reverse` starts at the end of the grid and moves
        backwards.
    reduction_method : {None, 'spear'}
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
    reduction_metric : {'val_acc'}
        Metric used to tune the reductions.
    x_val : ndarray
        User specified cross-validation data. (Default is None).
    y_val : ndarray
        User specified cross-validation labels. (Default is None).
    last_epoch_value : bool
        Set to True if the last epoch metric values are logged as opposed
        to the default which is peak epoch values for each round.
    print_params : bool
        Print params for each round on screen (useful when using TrainingLog
        callback for visualization)
    debug : bool
        Implements debugging feedback. (Default is False).

    """

    # TODO: refactor this so that we don't initialize global variables
    global self

    def __init__(self, x, y, params, model,
                 dataset_name=None, experiment_no=None,
                 x_val=None, y_val=None,
                 val_split=.3, shuffle=True,
                 round_limit=None,
                 grid_downsample=None,
                 random_method='uniform_mersenne',
                 seed=None,
                 search_method='random',
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
        self.params = params
        self.dataset_name = dataset_name
        self.experiment_no = experiment_no
        self.model = model
        self.x_val = x_val
        self.y_val = y_val
        self.val_split = val_split
        self.shuffle = shuffle
        self.random_method = random_method
        self.search_method = search_method
        self.reduction_method = reduction_method
        self.reduction_interval = reduction_interval
        self.reduction_window = reduction_window
        self.grid_downsample = grid_downsample
        self.reduction_threshold = reduction_threshold
        self.reduction_metric = reduction_metric
        self.reduce_loss = reduce_loss
        self.round_limit = round_limit
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
