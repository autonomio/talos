## Restore()

> The Deploy package can be read back to an object

```python
from talos import Restore

a = Restore(scan_object, 'experiment_name')
```

The `Deploy()` .zip package can be read back into a copy of the original experiment assets with `Restore()`. The object consists of:

- details of the scan
- model
- results of the experiment
- sample of x data
- sample of y data

### Restore Arguments

Parameter | Default | Description
--------- | ------- | -----------
`path_to_zip` | None | full path to the `Deploy` asset zip file
