import talos as ta

from talos.examples.datasets import iris
from talos.examples.models import iris_model
from talos.examples.params import iris as iris_params
from talos.utils.string_cols_to_numeric import string_cols_to_numeric


def iris_pipeline(round_limit=5):

    '''Performs a Scan with Iris dataset and simple dense net'''

    scan_object = ta.Scan(iris()[0],
                          iris()[1],
                          iris_params(),
                          iris_model,
                          round_limit=round_limit)

    scan_object.data = string_cols_to_numeric(scan_object.data)

    return scan_object
