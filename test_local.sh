export MPLBACKEND=agg

mprof run ./test/performance/memory_pressure.py
python ./test/performance/memory_pressure_check.py
python test_script.py
rm *.zip
rm *.csv
rm -rf 15*
rm ./test/*.csv
rm -rf testing*
rm -rf test_latest
rm mprofile_*.dat
rm -rf iris_test
rm ./test/*.log
