# Reference

Talos exposes parameter sweeps through the Python Scan pattern and through SFD/CLI experiments. This section describes public callables, defaults, result fields and failure boundaries. [Guides](../Guides/README.md) own end-to-end jobs; [Developer](../Developer/README.md) owns maintenance and documentation proof.

## Run a parameter sweep

| Interface | Use it for |
| --- | --- |
| [Scan](../Scan.md) | Five-argument model callbacks, candidate dictionaries, limits, result properties and persistence. |
| [Backends](../Backends.md) | Standalone Keras, TensorFlow/tf.keras and native Torch result/serialization contracts. |
| [Installation](../Install_Options.md) | Supported Python versions, framework extras and editable installation. |
| [SFD and CLI](../SFD_and_CLI.md) | Explicit model/preparation/parameter functions, YAML manifests and CLI commands. |

## Inspect and recover trained models

| Interface | Use it for |
| --- | --- |
| [Analyze](../Analyze.md) | Metric summaries, parameter tables, correlation and plots. |
| [Predict](../Predict.md) | Explicit candidate selection, raw inference and class conversion. |
| [Evaluate](../Evaluate.md) | Held-out F1 or MAE scores without retraining. |
| [Deploy](../Deploy.md) | Local ZIP packaging of a selected trained model and experiment provenance. |
| [Restore](../Restore.md) | Trusted native and historical archive restoration. |

## Construct a training callback

| Helper | Use it for |
| --- | --- |
| [Generator](../Generator.md) | Repeating array batches and framework Sequence batches. |
| [Hidden layers and shapes](../Hidden_Layers.md) | Dense/Dropout layer count and width candidates. |
| [Learning-rate normalizer](../Learning_Rate_Normalizer.md) | Fixed optimizer-specific scaling. |
| [Metrics](../Metrics.md) | Keras training metrics and their scientific units. |
| [Monitoring](../Monitoring.md) | Round progress, parameter printing, epoch logs and training plots. |
| [Energy draw](../Energy_Draw.md) | Endpoint GPU power samples and their watt-second estimate. |
| [Templates](../Templates.md) | Explicit dataset acquisition, preset parameters, model callbacks and pipelines. |

## Explore architecture presets

| Helper | Use it for |
| --- | --- |
| [AutoParams](../AutoParams.md) | Generate and narrow candidate dictionaries. |
| [AutoModel](../AutoModel.md) | Build a Keras training callback from architecture presets. |
| [AutoScan](../AutoScan.md) | Combine presets with the Scan execution interface. |
| [AutoPredict](../AutoPredict.md) | Score fitted candidates and predict with the held-out winner. |

Every executable fragment states its dependency and setup context. Framework extras are optional for the core, but required when a referenced operation needs that framework. The caller remains responsible for data suitability, training logic and an independent evaluation protocol.

## Read next

Run the [quickstart](../Guides/Quickstart.md), consult [Scan](../Scan.md) for an existing Talos callback, or use [migration](../Migration.md) to port it to an SFD.
