from ..scan.Scan import Scan


def Autom8(scan_object,x_val,y_val,metric='val_acc', n_models=10, to_csv=False):
    
    '''Pipeline automator
    
    Reduces the idea to prediction pipeline into a single 
    command where a Scan() process is followed by evaluating
    n best  
    
    Example use: 
    
    a = Autom8(X, Y, X, Y, p, diabetes_model, 'acc')
    
    to_csv :: Either 'False' or string for filename.

    '''

    # evaluate and add the evaluation scores
    scan_object.evaluate_models(scan_object,
                                x_val,
                                y_val,
                                metric=metric,
                                n=n_models)

    # make predictions with the best model
    preds = scan_object.best_model(scan_object, 'f1score_mean')
    scan_object.preds = preds.predict(x_val)

    # print out the best model parameters and stats 
    scan_object.preds_model = scan_object.data.sort_values('f1score_mean', ascending=False).iloc[0]

    if to_csv is not False:
      scan_object.data.to_csv(to_csv, index=False)
    
    return scan_object