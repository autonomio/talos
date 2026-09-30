from talos.commands.evaluate import evaluate_models
from talos.commands.predict import classes
from talos.backends import backend_for


def AutoPredict(scan_object, x_val, y_val, x_pred, task, metric='val_acc',
                n_models=10, folds=5, shuffle=True, asc=False, custom_objects=None,
                model_factory=None, average=None):
    evaluate_models(scan_object, x_val, y_val, task, n_models, metric, folds,
                    shuffle, asc, custom_objects=custom_objects,
                    model_factory=model_factory, average=average)
    continuous = task in ('continuous', 'regression')
    objective = 'eval_mae_mean' if continuous else 'eval_f1score_mean'
    model = scan_object.best_model(objective, asc=continuous, custom_objects=custom_objects)
    predictions = backend_for(model).predict(model, x_pred)
    scan_object.preds_model = model
    scan_object.preds_probabilities = predictions
    scan_object.preds_classes = classes(predictions, task)
    scan_object.preds_parameters = scan_object.data.sort_values(objective, ascending=continuous).iloc[0]
    return scan_object
