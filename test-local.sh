#!/bin/bash

export MPLBACKEND=agg

# run memory pressure test
mprof run ./tests/performance/memory_pressure.py
python3 ./tests/performance/memory_pressure_check.py

# run CI tests
python3 test-ci.py

# cleanup
rm mprofile_*.dat
rm ./test/*.log
rm ./test/*.csv
rm -rf test_*
rm -rf test