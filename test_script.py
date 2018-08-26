#!/usr/bin/env python

from test.core_tests.test_scan import TestIris, TestCancer
from test.core_tests.test_scan import TestReporting, TestLoadDatasets


if __name__ == '__main__':

    '''NOTE: test/core_tests/test_scan.py needs to be edited as well!'''

    # TODO describe what all this does
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
