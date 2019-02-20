#!/usr/bin/env python
import time

import talos as ta

from test.core_tests.test_scan import TestIris, TestCancer
from test.core_tests.test_scan import TestReporting, TestLoadDatasets
from test.core_tests.test_scan_object import test_scan_object


if __name__ == '__main__':

    '''NOTE: test/core_tests/test_scan.py needs to be edited as well!'''

    scan_object = test_scan_object()  # performs basic tests for scan_object

    start_time = str(time.strftime("%s"))

    ta.Autom8(scan_object, scan_object.x, scan_object.y)
    ta.Evaluate(scan_object)
    ta.Deploy(scan_object, start_time)
    ta.Restore(start_time + '.zip')

    TestCancer().test_scan_cancer_metric_reduction()
    TestCancer().test_scan_cancer_loss_reduction()
    TestCancer().test_linear_method()
    TestCancer().test_reverse_method()
    TestIris().test_scan_iris_explicit_validation_set()
    TestIris().test_scan_iris_explicit_validation_set_force_fail()
    TestIris().test_scan_iris_1()
    TestIris().test_scan_iris_2()
    TestReporting()
    TestLoadDatasets()
