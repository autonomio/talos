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


    # CORE ARGUMENTS
    ----------------
    x : ndarray
        1d or 2d array, or a list of arrays with features for the prediction
        task.
    y : ndarray
        1d or 2d array, or a list of arrays with labels for the prediction
        task.
    params : dict
        Lists all permutations of hyperparameters, a subset of which will be
        selected at random for training and evaluation.
    model : keras model
        Any Keras model with relevant declrations like params['first_neuron']
    experiment_name : str
        Experiment name will be used to produce a folder (unless already) it's
        there from previous iterations of the experiment. Logs of the
        experiment are saved in the folder with timestamp of start
        time as filenames.
    x_val : ndarray
        User specified cross-validation data. (Default is None).
    y_val : ndarray
        User specified cross-validation labels. (Default is None).
    val_split : float, optional
        The proportion of the input `x` which is set aside as the
        validation data. (Default is 0.3).

    # RANDOMNESS ARGUMENTS
    ----------------------

    random_method : str
        Determinines the way in which the grid_downsample is applied. The
        default setting is 'uniform_mersenne'.
    seed : int
        Sets numpy random seed.

    # LIMITER ARGUMENTS
    -------------------

    performance_target : None or list [metric, threshold, loss or not]
        Allows setting a threshold for a given metric, at which point the
        experiment will be concluded as successful.
        E.g. performance_target=['f1score', 0.8, False]
    fraction_limit : int
        The fraction of `params` that will be tested (Default is None).
        Previously grid_downsample.
    round_limit : int
        Limits the number of rounds (permutations) in the experiment.
    time_limit : None or str
        Allows setting a time when experiment will be completed. Use the format
        "%Y-%m-%d %H:%M" here.
    boolean_limit : None or lambda function
        Allows setting a limit to accepted permutations as a lambda function.
        E.g. example lambda p: p['first_neuron'] * p['hidden_layers'] < 220

    # OPTIMIZER ARGUMENTS
    ---------------------
    reduction_method : None or string
        If None, random search will be used as the optimization strategy.
        Otherwise use the name of the specific strategy, e.g. 'correlation'.
    reduction_interval : None or int
        The number of reduction method rounds that will be performed. (Default
        is None).
    reduction_window : None or int
        The number of rounds of the reduction method before observing the
        results. (Default is None).
    reduction_threshold: None or float
        The minimum value for reduction to be applied. For example, when
        the 'correlation' reducer finds correlation below the threshold,
        nothing is reduced.
    reduction_metric : None or str
        Metric used to tune the reductions. minimize_loss has to be set to True
        if this is a loss.
    minimize_loss : bool
        Must be set to True if a reduction_metric is a loss.

    # OUTPUT ARGUMENTS
    ------------------
    disable_progress_bar : bool
        Disable TQDM live progress bar.
    print_params : bool
        Print params for each round on screen (useful when using TrainingLog
        callback for visualization)

    # OTHER ARGUMENTS
    -----------------
    clear_session : bool
        If the backend session is cleared between every permutation.
    save_weights : bool
        If set to False, then model weights will not be saved and best_model
        and some other features will not work. Will reduce memory pressure
        on very large models and high number of rounds/permutations.
    """

    def __init__(self,
                 x,
                 y,
                 params,
                 model,
                 experiment_name,
                 x_val=None,
                 y_val=None,
                 val_split=.3,
                 random_method='uniform_mersenne',
                 seed=None,
                 performance_target=None,
                 fraction_limit=None,
                 round_limit=None,
                 time_limit=None,
                 boolean_limit=None,
                 reduction_method=None,
                 reduction_interval=50,
                 reduction_window=20,
                 reduction_threshold=0.2,
                 reduction_metric='val_acc',
                 minimize_loss=False,
                 disable_progress_bar=False,
                 print_params=False,
                 clear_session=True,
                 save_weights=True):

        self.x = x
        self.y = y
        self.params = params
        self.model = model
        self.experiment_name = experiment_name
        self.x_val = x_val
        self.y_val = y_val
        self.val_split = val_split

        # randomness
        self.random_method = random_method
        self.seed = seed

        # limiters
        self.performance_target = performance_target
        self.fraction_limit = fraction_limit
        self.round_limit = round_limit
        self.time_limit = time_limit
        self.boolean_limit = boolean_limit

        # optimization
        self.reduction_method = reduction_method
        self.reduction_interval = reduction_interval
        self.reduction_window = reduction_window
        self.reduction_threshold = reduction_threshold
        self.reduction_metric = reduction_metric
        self.minimize_loss = minimize_loss

        # display
        self.disable_progress_bar = disable_progress_bar
        self.print_params = print_params

        # performance
        self.clear_session = clear_session
        self.save_weights = save_weights
        # input parameters section ends

        # start runtime
        from .scan_run import scan_run
        scan_run(self)
