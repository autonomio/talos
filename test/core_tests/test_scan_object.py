# first load the pipeline
import talos as ta


def test_scan_object():

    print("Running Scan object test...")

    # the create the test based on it
    scan_object = ta.templates.pipelines.iris()
    keras_model = scan_object.best_model()
    scan_object.evaluate_models(x_val=scan_object.x,
                                y_val=scan_object.y)

    print("test_scan_object finished.")
    return scan_object
