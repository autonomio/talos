#!/usr/bin/env python

from test.core_tests.test_scan import TestIris, TestCancer, TestLoadDatasets


if __name__ == '__main__':
    TestIris().test_scan_iris_1()
    TestIris().test_scan_iris_2()
    TestCancer().test_scan_cancer()
    TestLoadDatasets()
