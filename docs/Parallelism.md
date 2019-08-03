# GPU Support

Talos supports scenarios where on a single system one or more GPUs are handling one or more simultaneous jobs. The base GPU support is handled by TensorFlow, so make sure that you have the GPU version of TensorFlow installed.

You can watch system GPU utilization anytime with:

`watch -n0.5 nvidia-smi`

## Single GPU, Single Job

Single GPU works out-of-the-box, as long as you have the GPU version of TensorFlow installed. In case you already have the CPU version installed, you have to uninstall TensorFlow and install the GPU version.

## Single GPU, Multiple Jobs

```python

from talos.utils.gpu_utils import parallel_gpu_jobs

# split GPU memory in two for two parallel jobs
parallel_gpu_jobs(0.5)
talos.Scan(...)

```
NOTE: The above lines must be run before the Scan() command:

A single GPU can be split to simultaneously perform several experiments. This is useful you want to work on more than one scope at one time, or when you're analyzing the results of an ongoing experiment with `Reporting()` and are ready to start the next experiment while keeping the first one running.

**NOTE:** GPU memory needs to be reserved pro-actively i.e. once the experiment is already running with full GPU memory, part of the memory can no longer be allocated to a new experiment.

## Multi-GPU, Single Job

```python
from talos.utils.gpu_utils import multi_gpu

# split a single job to multiple GPUs
model = multi_gpu(model)
model.compile(...)
```
NOTE: Include the above line in the input model before model.compile()

Multiple GPUs on a single machine can be assigned to work on a single machine in a parallelism fashion. This is useful when you have more than one GPU on a single machine, and want to speed up the experiment. Each GPU roughly speaking reduces compute time linearly.

## Force CPU

```python
from talos.utils.gpu_utils import force_cpu

# Force CPU use on a GPU system
force_cpu()
talos.Scan(...)
```
NOTE: Run the above lines before the Scan() command

Sometimes it's useful (for example when `batch_size` tends to be very small) to disable GPU and use CPU instead. This can be done simply by invoking `force_cpu()`.
