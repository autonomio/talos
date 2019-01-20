from ..scan.Scan import Scan


def Autom8(scan_object,
           x_val,
           y_val,
           n=10,
           metric='val_acc',
           folds=5,
           shuffle=True,
           average='binary',
           asc=False):

    '''Pipeline automator

    Reduces the idea to prediction pipeline into a single
    command where a Scan() process is followed by evaluating
    n best

    Example use:

    Parameters
    ----------
    scan_object : Scan() object
        A Scan() process needs to be completed first, and then the resulting
        object can be used as input here.
    x_val : ndarray
        Data to be used for 'x' in evaluation. Note that should be in the same
        format as the data which was used in the Scan() but not the same data.
    y_val : python dictionary
        Data to be used for 'y' in evaluation. Note that should be in the same
        format as the data which was used in the Scan() but not the same data.
    n : str
        Number of promising models to be included in the evaluation process.
        Time increase linearly with number of models.
    metric : str
        The metric to be used for deciding which models are promising.
        Basically the 'n' argument and 'metric' argument are combined to pick
        'n' best performing models based on 'metric'.
    folds : int
        Number of folds to be used in cross-validation.
    shuffle : bool
        If the data should be shuffled before cross-validation.
    average : str
        This parameter is required for multiclass/multilabel targets. If None,
        the scores for each class are returned. Otherwise, this determines
        the type of averaging performed on the data:

        'binary':
        Only report results for the class specified by pos_label.
        This is applicable only if targets (y_{true,pred}) are binary.

        'micro':
        Calculate metrics globally by counting the total true positives,
        false negatives and false positives.

        'macro':
        Calculate metrics for each label, and find their unweighted mean.
        This does not take label imbalance into account.

        'weighted':
        Calculate metrics for each label, and find their average weighted
        by support (the number of true instances for each label). This alters
        'macro' to account for label imbalance; it can result in an F-score
        that is not between precision and recall.

        'samples':
        Calculate metrics for each instance, and find their average
        (only meaningful for multilabel classification where this differs
        from accuracy_score).
    asc : bool
        This needs to be True for evaluation metrics that need to be minimized,
        and False when a metric needs to be maximized.

    '''

    # evaluate and add the evaluation scores
    scan_object.evaluate_models(x_val,
                                y_val,
                                n=n,
                                metric=metric,
                                folds=folds,
                                shuffle=shuffle,
                                average=average,
                                asc=False)

    # make predictions with the best model
    preds = scan_object.best_model('eval_f1score_mean')
    scan_object.preds = preds.predict(x_val)

    # print out the best model parameters and stats
    scan_object.preds_model = scan_object.data.sort_values('eval_f1score_mean',
                                                           ascending=False).iloc[0]

    return scan_object
