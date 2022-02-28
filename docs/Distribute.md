# Distribute()
An experiment can be split and distributed in multiple machines so as to make the experiment faster.

```python
from talos import DistributeScan
d=DistributeScan(params,config_path,file_path,destination_path)
d.distributed_run(run_local=run_local)
```

When you are distributing your script to multiple machines, the hyperparameters are split equally into different machines, and each machine runs with a different set of hyperparameter search. `Distribute()` takes in the parameters, and the file path with the talos experiment which needs to be distributed, along with configuration details of each of the machines for successful ssh connection. 

NOTE: The `distributed_run()` takes in a boolean argument of run_local, which helps run the distribution in the current machine as well. If set to false, distribution runs only in the machines mentioned in the config.

## Distribute Arguments

Parameter | type | Description
--------- | ------- | -----------
`params` | dict | a dictionary with the hyperparameter options
`config_path` | str | Path to config containing the machine details for ssh connection. Default: `config.json`
`file_path` | str | Path to the file containing the talos script to be distributed.Default: `script.py`
`destination_path` | str | Path to the file to be created in the destination machine: Default: `./temp.py`

## Distribute Package Contents

The distribute package consists of:

- distribute() method which can distribute the talos script/

