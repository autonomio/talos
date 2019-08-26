def AutoPredict(scan_object,
                x_val,
                y_val,
                x_pred,
                task,
                metric='val_acc',
                n_models=10,
                folds=5,
                shuffle=True,
                asc=False):

    '''Automatically handles the process of finding the best models from a
    completed `Scan()` experiment, evaluates those models, and uses the winner
    to make predictions on input data.

    NOTE: the input data must be in same format as 'x' that was
    used in `Scan()`.

    Parameters
    ----------
    scan_object : Scan() object
        A Scan() process needs to be completed first, and then the resulting
        object can be used as input here.
    x_val : ndarray or list of ndarray
        Data to be used for 'x' in evaluation. Note that should be in the same
        format as the data which was used in the Scan() but not the same data.
    y_val : ndarray or list of ndarray
        Data to be used for 'y' in evaluation. Note that should be in the same
        format as the data which was used in the Scan() but not the same data.
    y_pred : ndarray or list of ndarray
        Input data to be used for the actual predictions in evaluation. Note
        it should be in the same format as the data which was used in the
        Scan() but not the same data.
    task : string
        'binary', 'multi_class', 'multi_label', or 'continuous'.
    metric : str
        The metric to be used for deciding which models are promising.
        Basically the 'n' argument and 'metric' argument are combined to pick
        'n' best performing models based on 'metric'.
    n_models : str
        Number of promising models to be included in the evaluation process.
        Time increase linearly with number of models.
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
                                n_models=n_models,
                                task=task,
                                metric=metric,
                                folds=folds,
                                shuffle=shuffle,
                                asc=False)

    # get the best model based on evaluated score
    scan_object.preds_model = scan_object.best_model('eval_f1score_mean')

    # make predictions with the model
    scan_object.preds_probabilities = scan_object.preds_model.predict(x_pred)

    # make (class) predictiosn with the model
    scan_object.preds_classes = scan_object.preds_model.predict_classes(x_pred)

    # get the hyperparameter for the model
    scan_object.preds_parameters = scan_object.data.sort_values('eval_f1score_mean',
                                                           ascending=False).iloc[0]

    print(">> Added model, probabilities, classes, and parameters to scan_object")

    return scan_object
