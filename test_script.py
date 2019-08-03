#!/usr/bin/env python

if __name__ == '__main__':

    import talos
    from test.commands import *

    test_latest()
    test_autom8()
    test_templates()
    scan_object = test_scan()
    test_analyze(scan_object)
    test_random_methods()
    test_rest(scan_object)

    print("\n All tests successfully completed :) Good work. \n ")
