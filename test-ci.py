#!/usr/bin/env python

if __name__ == '__main__':

    from tests.commands import *

    scan_object = test_scan()
    test_latest()
    recover_best_model()
    test_random_methods()
    test_autom8()
    test_templates() 
    test_analyze(scan_object)
    test_lr_normalizer()
    test_predict()
    test_reducers()
    test_rest(scan_object)

    print("\n All tests successfully completed :) Good work. \n ")
