#!/usr/bin/env python
import time

import talos as ta

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

    # testing different model types
    from test.core_tests.test_scan import BinaryTest, MultiLabelTest

    BinaryTest().values_single_test()
    BinaryTest().values_list_test()
    BinaryTest().values_range_test()

    MultiLabelTest().values_single_test()
    MultiLabelTest().values_list_test()
    MultiLabelTest().values_range_test()

    # reporting specific testing
    from test.core_tests.test_scan import ReportingTest, DatasetTest

    ReportingTest()
    DatasetTest()

    # MOVE TO command specific tests

    # Scan() object tests
    scan_object = test_scan_object()

    # reporting tests
    test_reporting_object(scan_object)
    test_params_object()
    test_auto_scan()
    test_templates()

    # create a string for name of deploy file
    start_time = str(time.strftime("%s"))

    p = ta.Predict(scan_object)
    p.predict(scan_object.x)
    p.predict_classes(scan_object.x)

    ta.Autom8(scan_object, scan_object.x, scan_object.y)
    ta.Evaluate(scan_object)
    ta.Deploy(scan_object, start_time)
    ta.Restore(start_time + '.zip')

    test_random_methods()
    fit_generator = ta.utils.generator(scan_object.x, scan_object.y, 20)
    force_cpu()
