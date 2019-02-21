import talos as ta

from talos.examples.datasets import titanic
from talos.examples.params import titanic_params
from talos.examples.models import titanic_model

from talos.examples.datasets import iris
from talos.examples.models import iris_model
from talos.examples.params import iris as iris_params


def titanic_pipeline(round_limit=2, random_method='uniform_mersenne'):

    '''Performs a Scan with Iris dataset and simple dense net'''

    scan_object = ta.Scan(titanic()[0][:50],
                          titanic()[1][:50],
                          titanic_params(),
                          titanic_model,
                          round_limit=round_limit)

    return scan_object


def iris_pipeline(round_limit=5, random_method='uniform_mersenne'):

    '''Performs a Scan with Iris dataset and simple dense net'''

    scan_object = ta.Scan(iris()[0],
                          iris()[1],
                          iris_params(),
                          iris_model,
                          round_limit=round_limit)

    return scan_object
