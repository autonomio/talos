"""Expose manifest, parameter-sweep and recovery commands through Click."""

from pathlib import Path

import click

from talos import __version__


@click.group()
@click.version_option(version=__version__, prog_name='talos')
def cli() -> None:

    '''
    Talos — declarative ML experiment runner.

    \b
    Define experiments as YAML files and run them with a single command.
    Talos handles validation, compilation, parameter search, and result saving.

    \b
    Quick start:
      talos validate experiment.yaml       Check your YAML for errors
      talos profile experiment.yaml        Profile permutation space
      talos run experiment.yaml            Run the experiment
      talos run --dry-run experiment.yaml  Validate + compile only, no execution

    \b
    YAML structure:
      schema_version: "1.0"
      metadata:
        name: my_experiment
        mode: development
      sfd:
        module: my_sfd.py           # caller-owned params/prep/model
        params:
          epochs: [5, 10]
      uel:
        round_limit: 100
        search_strategy:
          type: random

    Caller SFD code owns data acquisition and training.
    Discover Keras, tf.keras and PyTorch examples with talos list-templates.
    '''


@cli.command()
@click.argument('yaml_file', type=click.Path(exists=True, path_type=Path))
@click.option('--parent', default=None, metavar='MANIFEST_ID',
              help='Parent manifest ID for lineage tracking (e.g. sha256:abc123...).')
@click.option('--message', '-m', default=None,
              help='Git commit message. Auto-generated if omitted.')
def commit(yaml_file: Path, parent: str | None, message: str | None) -> None:

    '''
    Validate and commit a YAML experiment file to the manifest store.

    \b
    Steps:
      1. Find the talos project root (walks up from the YAML file location)
      2. Validate the YAML file
      3. Content-address and store in manifests/committed/
      4. Update manifests/committed/index.json
      5. Git add + commit inside the project

    \b
    The committed file has a lineage block injected with its ID and timestamp.
    Committing the same file twice is a no-op (idempotent).

    \b
    Examples:
      talos commit manifests/first.yaml
      talos commit manifests/first.yaml --message "tuned parameters"
      talos commit manifests/first.yaml --parent sha256:abc123...
    '''

    from talos.cli.commands.commit import run_commit

    ok = run_commit(yaml_file, parent, message)
    raise SystemExit(0 if ok else 1)


@cli.command('ls')
def ls() -> None:

    '''
    List all committed manifests in the current project.

    \b
    Reads manifests/committed/index.json and displays each manifest
    with its short ID, name, and commit timestamp.

    \b
    Run from anywhere inside a Talos project directory.

    \b
    Examples:
      talos ls
    '''

    from talos.cli.commands.ls import run_ls

    ok = run_ls(Path.cwd())
    raise SystemExit(0 if ok else 1)


@cli.command()
@click.argument('yaml_file', type=click.Path(exists=True, path_type=Path))
def validate(yaml_file: Path) -> None:

    '''
    Validate a YAML experiment file.

    \b
    Checks:
      - YAML syntax and structure
      - Required fields (metadata, sfd, uel)
      - Schema version
      - Parameter lists are non-empty
      - Search strategy and output format values

    \b
    Exits 0 if valid, 1 if errors are found.

    \b
    Examples:
      talos validate experiment.yaml
      talos validate talos/yaml/templates/tf_keras.yaml
    '''

    from talos.cli.commands.validate import run_validate

    ok = run_validate(yaml_file)
    raise SystemExit(0 if ok else 1)


@cli.command('profile')
@click.argument('yaml_file', type=click.Path(exists=True, path_type=Path))
def profile_cmd(yaml_file: Path) -> None:

    '''
    Profile a YAML experiment — permutation space and runtime estimate.

    \b
    Always computed:
      - Total permutations and complexity rating (low / medium / high / extreme)
      - Per-parameter value counts

    \b
    Runtime sampling executes caller prep/model for a covering array of params.
    Samples report errors, elapsed time and NaN/Inf in caller-prepared numeric arrays.

    \b
    Exits 0 on success, 1 on validation or compilation failure.
    Sampling errors are non-fatal and reported in the output.

    \b
    Examples:
      talos profile experiment.yaml
      talos profile talos/yaml/templates/tf_keras.yaml
    '''

    from talos.cli.commands.profile import run_profile

    ok = run_profile(yaml_file)
    raise SystemExit(0 if ok else 1)


@cli.command()
@click.argument('target', required=False, default=None, metavar='[YAML_FILE | MANIFEST_URI]')
@click.option('--dry-run', is_flag=True, default=False,
              help='Validate and compile only — do not execute the experiment.')
@click.option('--resume', type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=None, metavar='RESULTS_DIR',
              help='Resume from a checkpoint directory instead of starting fresh.')
@click.option('--no-progress-bar', is_flag=True, default=False,
              help='Disable the experiment progress bar.')
def run(target: str | None, dry_run: bool, resume: Path | None, no_progress_bar: bool) -> None:

    '''
    Validate, compile, and run a YAML experiment file.

    \b
    Steps:
      1. Parse the YAML file
      2. Validate manifest structure
      3. Compile the manifest (resolve all callables)
      4. Build the parameter search domain
      5. Execute the native params/prep/model runner
      6. Save results to the configured output path

    \b
    Mode behaviour is controlled by metadata.mode in the YAML:
      development    Writes under ./results/dev/
      production     Writes under ./results/

    \b
    Output:
      Development mode:  ./results/dev/{name}_{datetime}/results.csv
      Production mode:   ./results/{name}_{datetime}/results.csv
      Committed manifest (manifest://):
                         ./results/[dev/]<short-hash>/<timestamp>/results.csv
      Set uel.output_format: parquet to also write results.parquet.
      Override the output path with uel.output_path in the YAML.

    \b
    Exits 0 on success, 1 on validation failure or runtime error.

    \b
    Examples:
      talos run experiment.yaml
      talos run --dry-run experiment.yaml
      talos run manifest://sha256:abc123ef...
      talos run --resume ./results/my_experiment_20260521_120000
      talos run --no-progress-bar experiment.yaml
    '''

    if resume is not None:
        if target is not None:
            click.secho('Cannot specify both a YAML file and --resume.', fg='red')
            raise SystemExit(1)
        if dry_run:
            click.secho('--dry-run has no effect with --resume.', fg='red')
            raise SystemExit(1)
        from talos.cli.commands.resume import run_resume

        ok = run_resume(resume, progress_bar=not no_progress_bar)
    elif target is not None:
        from talos.cli.commands.run import run_experiment

        yaml_path, manifest_id, results_base = _resolve_target(target)
        if yaml_path is None:
            raise SystemExit(1)
        ok = run_experiment(yaml_path, dry_run=dry_run, manifest_id=manifest_id, results_base=results_base, progress_bar=not no_progress_bar)
    else:
        click.secho('Provide a YAML file or --resume <results-dir>.', fg='red')
        raise SystemExit(1)

    raise SystemExit(0 if ok else 1)


def _resolve_target(target: str) -> tuple[Path | None, str | None, Path]:
    from talos.yaml.store import (
        MANIFEST_URI_SCHEME,
        SHA256_PREFIX,
        normalize_manifest_ref,
        resolve_manifest_uri,
    )

    if target.startswith((MANIFEST_URI_SCHEME, SHA256_PREFIX)):
        uri = normalize_manifest_ref(target)
        try:
            path, project_root = resolve_manifest_uri(uri, Path.cwd())
        except ValueError as exc:
            click.secho(f'  ✗ {exc}', fg='red')
            return None, None, Path('.')
        click.echo(f"  Resolved {uri}")
        return path, path.stem, project_root

    p = Path(target)
    if not p.exists():
        click.secho(f"  ✗ File not found: {target}", fg='red')
        return None, None, Path('.')
    return p, None, Path('.')


@cli.command('list-templates')
def list_templates() -> None:

    '''
    List all available YAML experiment templates.

    \b
    Examples:
      talos list-templates
    '''

    from talos.cli.commands.list_templates import run_list_templates

    run_list_templates()


@cli.command()
@click.argument('output', type=click.Path(path_type=Path))
@click.option('--template', default=None, help='Template name to scaffold from (e.g. tf_keras).')
def init(output: Path, template: str | None) -> None:

    '''
    Scaffold a new YAML experiment file from a template.

    \b
    Copies the selected template to OUTPUT and sets metadata.name
    to the output filename stem.

    \b
    When run from inside a Talos project (talos.toml present), a bare name
    is placed in manifests/ automatically. The .yaml extension is optional.

    \b
    Examples:
      talos init my_experiment --template tf_keras
      talos init my_experiment.yaml --template tf_keras
      talos init path/to/my_experiment.yaml --template tf_keras
      talos init my_experiment               # prompts for template, default tf_keras
    '''

    from talos.cli.commands.init import run_init

    ok = run_init(output, template)
    raise SystemExit(0 if ok else 1)


@cli.command()
def reindex() -> None:

    '''
    Rebuild the manifest store index from the committed manifest files.

    \b
    The committed manifests are the source of truth; index.json is a derived
    cache. Use this to repair the index after a bad merge, a manual edit, or a
    pull that left index.json out of sync with manifests/committed/.

    \b
    Only rewrites index.json — the committed manifests are never modified.

    \b
    Examples:
      talos reindex
    '''

    from talos.cli.commands.reindex import run_reindex

    ok = run_reindex(Path.cwd())
    raise SystemExit(0 if ok else 1)


@cli.command()
@click.argument('manifest_ref', metavar='MANIFEST_ID')
@click.argument('name', required=False, default=None)
def fork(manifest_ref: str, name: str | None) -> None:

    '''
    Fork a committed manifest into a new working file.

    \b
    Copies a committed manifest into manifests/ as a development-mode working
    file, records the source as its lineage parent, and lets you iterate on it.
    The parent lineage is committed automatically on the next talos commit.

    \b
    MANIFEST_ID accepts a bare hash, sha256:<hash>, or manifest://sha256:<hash>,
    and short hashes are resolved when unambiguous.

    \b
    Examples:
      talos fork sha256:d3a5d334
      talos fork sha256:d3a5d334 my_tuned_run
      talos fork manifest://sha256:d3a5d334
    '''

    from talos.cli.commands.fork import run_fork

    ok = run_fork(manifest_ref, name, Path.cwd())
    raise SystemExit(0 if ok else 1)


@cli.command()
@click.argument('manifest_ref', metavar='MANIFEST_ID')
def lineage(manifest_ref: str) -> None:

    '''
    Show the lineage chain for a committed manifest.

    \b
    Walks the parent chain from the given manifest up to its root and prints
    it root-first, with the target manifest marked.

    \b
    MANIFEST_ID accepts a bare hash, sha256:<hash>, or manifest://sha256:<hash>,
    and short hashes are resolved when unambiguous.

    \b
    Examples:
      talos lineage sha256:37222f71
      talos lineage 37222f71
    '''

    from talos.cli.commands.lineage import run_lineage

    ok = run_lineage(manifest_ref, Path.cwd())
    raise SystemExit(0 if ok else 1)


@cli.command()
@click.argument('project_name')
@click.option('--backup-remote', default=None,
              help='Git remote URL for manifest store backup (can also be set interactively).')
@click.option('--from', 'from_remote', default=None, metavar='REMOTE_URL',
              help='Restore a project from a backup remote instead of the template.')
def new(project_name: str, backup_remote: str | None, from_remote: str | None) -> None:

    '''
    Create a new Talos project from local templates.

    \b
    Creates local project files and optionally configures a backup remote.
    With --from, the project is restored from a backup remote with its full
    history instead of scaffolded from the template.

    \b
    Examples:
      talos new my-project
      talos new my-project --backup-remote git@github.com:user/my-project.git
      talos new my-project --from git@github.com:user/my-project.git
    '''

    if from_remote is not None and backup_remote is not None:
        click.secho('  ✗ Cannot combine --from with --backup-remote.', fg='red')
        raise SystemExit(1)

    from talos.cli.commands.new import run_new

    ok = run_new(project_name, backup_remote, from_remote)
    raise SystemExit(0 if ok else 1)


@cli.command()
def backup() -> None:

    '''
    Snapshot the project and push it to the configured backup remote.

    \b
    Stages and commits the project's current state (honoring .gitignore, so
    development runs under results/dev/ stay local), then pushes to the remote.
    Reads backup_remote from talos.toml. Set it first (via talos new or by
    editing talos.toml).

    \b
    Restore on another machine with: talos new <name> --from <remote-url>

    \b
    Examples:
      talos backup
    '''

    from talos.cli.commands.backup import run_backup

    ok = run_backup(Path.cwd())
    raise SystemExit(0 if ok else 1)
