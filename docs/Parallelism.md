# GPU Support

Talos supports scenarios where on a single system one or more GPUs are handling one or more simultaneous jobs. GPU execution is handled by the selected TensorFlow, Keras or Torch backend. Install a backend build compatible with your device and drivers; see [Backends](Backends.md). The examples are safe on CPU-only systems, where no GPU is configured.

You can watch system GPU utilization anytime with:

`nvidia-smi` (NVIDIA hardware only; not required for the CPU examples).

## Single GPU, Single Job

A supported visible GPU is used by the backend according to its device configuration. Verify the backend/driver compatibility before running; no universal CPU/GPU package replacement command applies across platforms.

## Single GPU, Multiple Jobs

The GPU configuration fragments below require the optional TensorFlow backend, including when checking its CPU fallback. Fractional memory limits rely on NVIDIA tooling; other accelerator backends need their own memory configuration.

Examples below use the held-out Iris setup in [Scan → Minimal Example](Scan.md#minimal-example). Run that setup first; it defines `scan_object`, `p`, `input_model`, `x`, `y`, `x_val`, `y_val`, `x_test`, and `y_test`.

```python
from talos.utils.gpu_utils import parallel_gpu_jobs

# split GPU memory in two for two parallel jobs
parallel_gpu_jobs(0.5)
gpu_scan = talos.Scan(x, y, p, input_model, 'gpu_jobs', x_val=x_val, y_val=y_val,
                      seed=17, disable_progress_bar=True)

```
NOTE: The above lines must be run before the Scan() command:

A single GPU can be split to simultaneously perform several experiments. This is useful you want to work on more than one scope at one time, or when you're analyzing the results of an ongoing experiment with `Reporting()` and are ready to start the next experiment while keeping the first one running.

**NOTE:** GPU memory needs to be reserved pro-actively i.e. once the experiment is already running with full GPU memory, part of the memory can no longer be allocated to a new experiment.

## Multi-GPU, Single Job

```python
from talos.utils.gpu_utils import multi_gpu

# split a single job to multiple GPUs
model = keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(3, activation='softmax')])
model = multi_gpu(model)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
```
NOTE: Include the above line in the input model before model.compile()

Multiple GPUs on a single machine can be assigned to work on a single machine in a parallelism fashion. This is useful when you have more than one GPU on a single machine, and want to speed up the experiment. Speedup depends on model size, batch size, device throughput and communication. A CPU-only run verifies the fallback, not multi-GPU execution.

## Force CPU

```python
from talos.utils.gpu_utils import force_cpu

# Force CPU use on a GPU system
force_cpu()
cpu_scan = talos.Scan(x, y, p, input_model, 'cpu', x_val=x_val, y_val=y_val,
                      seed=17, disable_progress_bar=True)
```
NOTE: Run the above lines before the Scan() command

Sometimes it's useful (for example when `batch_size` tends to be very small) to disable GPU and use CPU instead. This can be done simply by invoking `force_cpu()`.

For backend-independent sweep parallelism, distribute pending parameter rows to independent workers. Each worker owns its process/run directory; trial identities include a stable shard namespace.

```python
from talos.parameters.DistributeParamSpace import DistributeParamSpace
shards = DistributeParamSpace(p, machines=2, seed=17).param_spaces
worker_scans = [talos.Scan(x, y, shard, input_model, f'worker_{worker}',
                          x_val=x_val, y_val=y_val, seed=17, disable_progress_bar=True)
                for worker, shard in shards.items()]
```
This bounded example runs workers sequentially on CPU. Launch them in separate processes to run simultaneously; the core does not automatically start worker processes.
