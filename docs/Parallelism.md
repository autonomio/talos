# Devices and parallel workers

Talos supports scenarios where on a single system one or more GPUs are handling one or more simultaneous jobs. GPU execution is handled by the selected TensorFlow, Keras or Torch backend. Install a backend build compatible with your device and drivers; see [Backends](Backends.md). The examples are safe on CPU-only systems, where no GPU is configured.

## Prerequisites

Have a working callback and a compatible [backend environment](Backends.md). The device helpers below use TensorFlow; native Torch device placement remains callback code. GPU runs additionally need compatible hardware, drivers and the backend’s accelerator support. CPU-only systems can run the fallback examples. Use a writable, separate run directory for each worker.

1. Choose whether to distribute one model across devices or split parameter trials across processes.
2. Configure device visibility and memory before training starts.
3. Run the relevant example below using the linked [Scan setup](Scan.md#minimal-example).
4. Check recorded trial rows and actual device utilization in the environment you intend to use.

You can watch system GPU utilization with:

`nvidia-smi` (NVIDIA hardware only; not required for the CPU examples).

## One GPU, one job

A supported visible GPU is used by the backend according to its device configuration. Verify the backend/driver compatibility before running; no universal CPU/GPU package replacement command applies across platforms.

## One GPU, multiple jobs

The GPU configuration fragments below require the optional TensorFlow backend, including when checking its CPU fallback. Fractional memory limits rely on NVIDIA tooling; other accelerator backends need their own memory configuration.

Examples below use the held-out Iris setup in [Scan → Minimal Example](Scan.md#minimal-example). Run that setup first; it defines `scan_object`, `p`, `input_model`, `x`, `y`, `x_val`, `y_val`, `x_test`, and `y_test`.

```python
from talos.utils.gpu_utils import parallel_gpu_jobs

# split GPU memory in two for two parallel jobs
parallel_gpu_jobs(0.5)
gpu_scan = talos.Scan(x, y, p, input_model, 'gpu_jobs', x_val=x_val, y_val=y_val,
                      seed=17, disable_progress_bar=True)

```

Run the memory configuration before starting the scan and before the framework initializes its device runtime.

A single GPU can host several independently started experiments when each process has enough device memory. This is useful when exploring separate model scopes or starting a new experiment after inspecting an ongoing run’s recorded results. The helper configures memory; it does not start another process.

Reserve GPU memory before starting training. A framework may reject memory or device changes after runtime initialization; start a fresh process when reconfiguring an initialized device.

## Multiple GPUs, one job

```python
from talos.utils.gpu_utils import multi_gpu

# split a single job to multiple GPUs
model = keras.Sequential([keras.Input(shape=(4,)), keras.layers.Dense(3, activation='softmax')])
model = multi_gpu(model)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
```

Place the model-distribution call in the input callback before compiling the returned model.

Multiple GPUs on a single machine can execute one model through the framework’s distribution mechanism. This is useful when you have more than one GPU on a single machine, and want to speed up the experiment. Speedup depends on model size, batch size, device throughput and communication. A CPU-only run verifies the fallback, not multi-GPU execution.

## Force CPU execution

```python
from talos.utils.gpu_utils import force_cpu

# Force CPU use on a GPU system
force_cpu()
cpu_scan = talos.Scan(x, y, p, input_model, 'cpu', x_val=x_val, y_val=y_val,
                      seed=17, disable_progress_bar=True)
```

Select CPU execution before the scan and before TensorFlow initializes GPU devices.

Sometimes it's useful (for example when `batch_size` tends to be very small) to disable GPU and use CPU instead. This can be done simply by invoking `force_cpu()`.

## Split a sweep into worker shards

For backend-independent sweep parallelism, distribute pending parameter rows to independent workers. Each worker owns its process/run directory; trial identities include a stable shard namespace.

```python
from talos.parameters.DistributeParamSpace import DistributeParamSpace
shards = DistributeParamSpace(p, machines=2, seed=17).param_spaces
worker_scans = [talos.Scan(x, y, shard, input_model, f'worker_{worker}',
                          x_val=x_val, y_val=y_val, seed=17, disable_progress_bar=True)
                for worker, shard in shards.items()]
```

This bounded example runs workers sequentially on CPU. Launch them in separate processes to run simultaneously; the core does not automatically start worker processes.

## Expected result and failure boundaries

Each device example returns a completed Scan result. On a CPU-only system, the TensorFlow helpers have no visible GPU to configure, and `multi_gpu` returns the model unchanged when fewer than two logical GPUs are available. That proves fallback behavior rather than physical GPU distribution.

The sharding example produces two independent Scan results whose candidate rows come from the same parameter space. Worker namespaces distinguish trial identities. It runs the workers sequentially as written; separate processes and separate run directories are required for concurrent execution. Talos does not schedule a cluster or merge worker results automatically.

Fractional TensorFlow memory configuration calls `nvidia-smi` when a GPU is present; missing tooling, driver incompatibility or an initialized device can fail at that boundary. Torch helpers do not automatically move caller data or models to a chosen device. Verify memory usage, throughput and physical-device behavior separately from the CPU documentation checks.

## Read next

Review [backend compatibility](Backends.md), [Scan](Scan.md) for worker inputs, and [maintenance verification](Maintenance.md) for the boundary between CPU evidence and accelerator evidence.
