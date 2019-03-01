#!/usr/bin/env python
import time

import talos as ta

from test.core_tests.test_scan import TestIris, TestCancer
from test.core_tests.test_scan import TestReporting, TestLoadDatasets
from test.core_tests.test_scan_object import test_scan_object
from test.core_tests.test_reporting_object import test_reporting_object
from test.core_tests.test_random_methods import test_random_methods
from test.core_tests.test_params_object import test_params_object
from test.core_tests.test_auto_scan import test_auto_scan
from test.core_tests.test_templates import test_templates

from talos.utils.generator import generator
from talos.utils.gpu_utils import force_cpu


if __name__ == '__main__':

    '''NOTE: test/core_tests/test_scan.py needs to be edited as well!'''

    # Scan
    scan_object = test_scan_object()

    # Reporting
    test_reporting_object(scan_object)
    test_params_object()
    test_auto_scan()
    test_templates()

    start_time = str(time.strftime("%s"))

    p = ta.Predict(scan_object)
    p.predict(scan_object.x)
    p.predict_classes(scan_object.x)

    ta.Autom8(scan_object, scan_object.x, scan_object.y)
    ta.Evaluate(scan_object)
    ta.Deploy(scan_object, start_time)
    ta.Restore(start_time + '.zip')

    test_random_methods()
    fit_generator = generator(scan_object.x, scan_object.y, 20)
    force_cpu()

    TestCancer().test_scan_cancer_metric_reduction()
    TestCancer().test_scan_cancer_loss_reduction()
    TestCancer().test_linear_method()
    TestCancer().test_reverse_method()
    TestIris().test_scan_iris_explicit_validation_set()
    TestIris().test_scan_iris_explicit_validation_set_force_fail()
    TestIris().test_scan_iris_1()
    TestIris().test_scan_iris_2()
    TestIris().test_scan_iris_3()
    TestReporting()
    TestLoadDatasets()
