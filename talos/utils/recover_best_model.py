

def recover_best_model(x_train,
                       y_train,
                       x_val,
                       y_val,
                       experiment_log,
                       input_model,
                       x_cross=None,
                       y_cross=None,
                       n_models=5,
                       task='multi_label'):

    '''Recover best models from Talos experiment log.

    x_train | array | same as was used in the experiment
    y_train | array | same as was used in the experiment
    x_val | array | same as was used in the experiment
    y_val | array | same as was used in the experiment
    x_cross | array | data for the cross-validation or None for use x_val
    y_cross | array | data for the cross-validation or None for use y_val
    experiment_log | str | path to the Talos experiment log
    input_model | function | model used in the experiment
    n_models | int | number of models to cross-validate
    task | str | binary, multi_class, multi_label or continuous

    Returns a pandas dataframe with the cross-validation results
    and the models.

    '''

    import pandas as pd
    import sklearn as sk
    import numpy as np

    from talos.utils.validation_split import kfold

    # read the experiment log into a dataframe
    df = pd.read_csv(experiment_log)

    # handle input data scenarios

    if x_cross is None or y_cross is None:
        x_cross = x_val
        y_cross = y_val

    # for final output
    results = []
    models = []

    for i in range(n_models):

        # get the params for the model and train it
        params = df.sort_values('val_acc', ascending=False).drop('val_acc', 1).iloc[i].to_dict()
        history, model = input_model(x_train, y_train, x_val, y_val, params)

        # start kfold cross-validation
        out = []
        folds = 5
        kx, ky = kfold(x_cross, y_cross, folds, True)

        for i in range(folds):

            y_pred = model.predict(kx[i]).argmax(axis=1)

            if task == 'binary':
                y_pred = y_pred >= .5
                scores = sk.metrics.f1_score(y_pred, ky[i], average='binary')

            elif task == 'multi_class':
                y_pred = y_pred.argmax(axis=-1)
                scores = sk.metrics.f1_score(y_pred, ky[i], average='macro')

            if task == 'multi_label':
                y_pred = model.predict(kx[i]).argmax(axis=1)
                scores = sk.metrics.f1_score(y_pred,
                                             ky[i].argmax(axis=1),
                                             average='macro')

            elif task == 'continuous':
                y_pred = model.predict(kx[i])
                scores = sk.metrics.mean_absolute_error(y_pred, ky[i])

            out.append(scores)

        results.append(np.mean(out))
        models.append(model)

    out = df.sort_values('val_acc', ascending=False).head(n_models)
    out['crossval_mean_f1score'] = results

    return out, models
