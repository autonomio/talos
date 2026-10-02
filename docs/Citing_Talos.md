# Cite Talos and identify the experiment

Cite the software version that produced your results and retain the experiment
record with your research artifacts. This guide covers software attribution
and the identifiers needed to find a Talos run. Model construction, data
acquisition and scientific evaluation remain your code.

The maintainer reports Talos use in at least 1,000 research papers. If Talos is
part of your work, a version-specific citation helps the next researcher find
the software and experiment you used.

## Before you begin

For a software citation, you need the version you used and its source or release
URL. For a development checkout, also retain the full Talos source commit. To
identify an experiment, keep its run directory and `metadata.json`.

No code execution or additional dependency is required to read the software
citation files. Run metadata is available after a completed or checkpointed
[parameter sweep](Guides/Quickstart.md).

See [installation](Install_Options.md) for supported environments,
[Scan](Scan.md) for the established Python interface, and
[SFD and CLI](SFD_and_CLI.md) for file-based experiments. The
[migration guide](Migration.md) explains changed behavior across Talos versions.

## Choose the software citation

1. Open the [Talos repository](https://github.com/autonomio/talos) and choose
   **Cite this repository**. GitHub reads [CITATION.cff](../CITATION.cff) and
   supplies APA and BibTeX formats. A downloadable [BibTeX file](../CITATION.bib)
   carries the same software metadata. Its standard `@misc` entry includes the version in the note so traditional bibliography styles retain it.
2. Check that the citation identifies the version you actually used. If you
   used an earlier release, use that release's citation information and retain
   its source or release permalink.
3. For development code, include the full source commit and a permalink to that
   commit. Record changes made after checkout alongside the source you used.

The current source version is 2.0.4 and is unreleased. Its citation metadata
contains no release date or DOI. A later release must update the version and
actual release metadata together. Use the version in the recorded experiment,
even if your current environment has since been upgraded.

## Identify the recorded run

Talos stores these identifiers and settings in the run directory:

| Research item | Recorded location |
| --- | --- |
| Talos, framework and Python versions | `metadata.json` → `environment` |
| Run directory identifier | The directory name exposed by `scan.run_dir` or `result.run_dir` |
| Run configuration and data identity | `metadata.json` → `identity_hash` and `identity` |
| Manifest content identifier, when used | `metadata.json` → `yaml_reference.manifest_id` |
| Caller model or SFD source | `metadata.json` → `sfd`, `identity.model`, `identity.prep` and `source_bundle` |
| Candidate settings, search policy and seed | `metadata.json` → `identity.params`, `identity.strategy` and `identity.seed` |
| Metric and direction | `metadata.json` → `objective` |
| Observed trial records | `results.csv` and `round_data.jsonl` |

The run directory name is a local identifier; it becomes useful to another
researcher when you retain or publish the corresponding artifacts. The
`identity_hash` identifies the recorded configuration and inputs. It does not
summarize the measured performance or replace a citation to the software.

A Python Scan run can have no manifest, so its `manifest_id` is absent. Preserve
its caller callback and parameter dictionary. For a manifest run, retain the
committed YAML and the caller SFD beside the run records. The manifest stores
configuration; your caller code owns data acquisition, preparation and training.

## Keep the machine-readable records

[CITATION.cff](../CITATION.cff) owns software citation metadata. The generated
[BibTeX file](../CITATION.bib) carries the same author, title, version, source
URL and license. These files are kept in sync with the package version and
changelog; use the files corresponding to the software you used.

For the experiment, retain `metadata.json` together with the manifest, source
bundle, trial records and model artifacts. Its recorded environment,
`identity_hash` and optional `manifest_id` let a later reader identify the run
without reconstructing a citation from your current installation. Data
fingerprints and source identifiers remain part of that record.

Existing run metadata does not establish the exact Talos Git revision. Record
that revision from the software installation you used. A `git_revision` on a
caller model or SFD identifies the caller repository; use it for that code's
provenance. Keep both revisions when Talos and the experiment code come from
separate repositories.

## Report the research context

In the methods or supplementary material, include the Talos version and source
revision, framework versions, caller model or SFD, parameter space, search
policy, seed, objective metric and direction, and how the final candidate was
selected. Identify the data source, preparation and train/validation/test split
with enough detail to reconstruct them.

Talos records data fingerprints; they do not supply a dataset citation or the
scientific meaning of a split. Cite the dataset or acquisition method using
its actual provenance. Retain the caller source that made the split and applied
preprocessing. Report any independent held-out evaluation separately from
validation measurements used during the sweep.

A seed and matching software environment define part of the experiment.
Hardware, framework kernels and external services can still affect the result.
Keep the recorded environment and state these execution conditions when they
matter to your analysis.

## Historical citation

The established README at
[commit 715d2b6](https://github.com/autonomio/talos/blob/715d2b6477c775d0444dbaa88a37624d4577e07b/README.md#loudspeaker-citations)
requested this citation:

> Autonomio Talos [Computer software]. (2024). Retrieved from <http://github.com/autonomio/talos>.

That text remains available for earlier work. It contains a citation year;
it does not identify an exact Talos release. [Talos 1.4](https://pypi.org/project/talos/1.4/)
was published on 21 April 2024 and corresponds to the
[v1.4 release](https://github.com/autonomio/talos/releases/tag/v1.4).
Include the version actually used rather than assigning 1.4 to all work dated 2024.

Earlier README citation variants used
[2018](https://github.com/autonomio/talos/blob/a9fbe3550af3511ff53b51e3327ec9f090e46849/README.md#citations),
[2019](https://github.com/autonomio/talos/blob/21452f07b281017c7035ac4a84a011b1b82b170e/README.md#loudspeaker-citations)
and [2020](https://github.com/autonomio/talos/blob/7da4983b47a3f7d0c464a5c6c8ed4828478155c5/README.md#loudspeaker-citations).

## If citation details are missing

Use the retained software version and source URL. Add a full source revision
when your records establish it. Leave unknown release dates and identifiers
unasserted. The project citation files do not establish a DOI or a paper
bibliography. Report missing provenance as a limitation of the research record.

## Read next

Inspect [Scan outputs](Scan.md), retain a reproducible
[SFD and manifest run](SFD_and_CLI.md), or check
[migration and recovery](Migration.md) before comparing results across versions.

GitHub's [citation-file documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files)
and the [CFF schema guide](https://github.com/citation-file-format/citation-file-format/blob/main/schema-guide.md)
define the supported software citation formats and metadata fields.
