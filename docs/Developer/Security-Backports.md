# Owned dependency security backports

The legacy TensorFlow lane preserves old Talos callbacks with TensorFlow 2.14.1,
Keras `2.14.0+autonomio.1` and Protobuf `4.25.9+autonomio.1`. These are Autonomio
backports, not upstream releases. Modern extras remain the default for new work.
Use Python 3.10.12+ or 3.11.4+ for the legacy lane.

## Reconstruct and install

From an exact reviewed Talos source checkout, reconstruct the two pure-Python
wheels without importing a training framework:

```sh
python -m tools.security.wheels --cache build/legacy-wheel-cache --output build/legacy-security
```

The builder downloads the official PyPI artifacts identified in
`tools/security/legacy-backports.json`, verifies every original RECORD member,
applies exact source patches and checks the complete installed-tree and final
wheel digests. Fixed member order, timestamps and stored ZIP members make the
output independent of host compression libraries. `receipts.json` records the
artifact identities. A mismatched cache, patch, output or source tree fails.

In a fresh legacy Python environment, install the two generated wheels with
`python -m pip install --no-deps build/legacy-security/*.whl`, then install
`python -m pip install 'tensorflow==2.14.1' 'numpy==1.26.4'` and select either
`python -m pip install -e '.[legacy-tensorflow]'` for Talos 2 or
`python -m pip install 'talos==1.4' 'ipython<9'` for the unchanged old generation.
Run `python -m pip check` before training. Preserve the wheels and receipts
with the research environment. Signed downloadable assets exist only after a
release completes the [release policy](Release-Policy.md); candidate builds
are not signed releases.

## Repairs and compatibility

| Surface | Repair and verification |
| --- | --- |
| Keras deserialization | Safe Lambda defaults; safe scope covers config; restrict implicit imports and function reexports; retain explicit custom objects |
| Vocabulary assets | Default config deserialization to a safe scope; retain authored construction and explicitly trusted loading; save embedded assets and restore after original deletion |
| HDF5 | Reject links, virtual/external datasets and excessive allocations before data reads |
| ZIP and tar | Contain paths and extraction; bound expansion; reject archive links without deleting existing destinations |
| NPZ | Disable object-array pickle in safe mode; retain explicit trusted loading |
| Protobuf Any | Account for recursion in nested Any JSON conversion |

`governance/tests/test_legacy_backports.py` exercises the repaired packages,
a genuine original Keras 2.14 artifact, real Iris predictions, registered custom
layers, recurrent models and optimizer continuation. Private extraction helpers
with unsafe current-directory/prefix logic are removed. Public `get_file`
extraction remains tested. [Migration](../Migration.md#legacy-security-backports)
records deliberate loading restrictions and environment/resume boundaries.

## Audit and maintenance

The existing legacy CI job installs the exact hash lock, verifies every owned
installed member independently of its mutable RECORD, and audits the complete
graph using the original upstream versions. It retains the raw findings,
lookup identities and dispositions. `tools/security/legacy-advisories.json`
binds each known finding to repair tests or absent Keras 3-only modules.
New IDs, missing dependencies, altered sources and auditor errors fail.
No package rename or broad vulnerability ignore removes advisory evidence.
The single-job `dependency_monitor.yml` repeats the legacy upstream lookup and
documentation audit every Thursday at 02:20 UTC, after the three staggered
Dependabot update days. It has a 15-minute timeout and no pull-request trigger.
Failures retain audit evidence; new fixes require a reviewed PR.

The documentation toolchain applies source-bound patches to every installed
`braces` 3.0.3 and `http-cache-semantics` 4.2.0 copy during `npm ci`. Tests reject
adversarial nesting and shared-cache authorization bypass while preserving
ordinary glob and cache behavior. The audit independently verifies patched
bytes. Original npm findings and existing approval expiry remain visible;
see [documentation dependency exceptions](Documentation-System.md#documentation-dependency-exceptions).

Review upstream fixes and new advisories on each dependency update. Replace
owned patches with a compatible upstream release when possible, rerun original
model/recovery and patch regression checks, regenerate hash locks and record
changed identities. Preserve upstream licenses and attribution in each wheel;
[third-party notices](../../THIRD_PARTY.md) record ownership.

## Read next

- [Installation](../Install_Options.md)
- [Maintenance](../Maintenance.md)
- [Release policy](Release-Policy.md)
