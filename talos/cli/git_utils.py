import shutil
import subprocess
from pathlib import Path


def git_executable() -> str:

    '''
    Locate the git binary on PATH.

    Returns:
        str: Absolute path to the git executable

    Raises:
        FileNotFoundError: If git is not found on PATH

    '''

    git = shutil.which('git')
    if git is None:
        raise FileNotFoundError('git not found on PATH')
    return git


def run_git(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:

    '''
    Run a git subcommand with captured text output and no exception on failure.

    Args:
        args (list[str]): Git arguments, e.g. ['push', remote_url, 'HEAD']
        cwd (Path | None): Working directory to run git in

    Returns:
        subprocess.CompletedProcess: The completed process; inspect returncode
            and stderr for the result

    Raises:
        FileNotFoundError: If git is not found on PATH

    '''

    return subprocess.run(
        [git_executable(), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def git_add_and_commit(repo_root: Path, path: Path, message: str) -> bool:

    '''
    Stage a path and create a git commit inside a repository.

    Args:
        repo_root (Path): Root directory of the git repository
        path (Path): Path to stage (relative to repo_root)
        message (str): Commit message

    Returns:
        bool: True if both git add and git commit succeeded

    '''

    try:
        add = run_git(['add', str(path)], cwd=repo_root)
    except FileNotFoundError:
        return False
    if add.returncode != 0:
        return False
    commit = run_git(['commit', '-m', message], cwd=repo_root)
    return commit.returncode == 0


def git_push(repo_root: Path, remote_url: str) -> tuple[bool, str]:

    '''
    Push the current branch of a repository to a remote URL.

    Args:
        repo_root (Path): Root directory of the git repository
        remote_url (str): Destination git remote URL

    Returns:
        tuple[bool, str]: (succeeded, stderr) where stderr carries git's
            error output when the push fails

    Raises:
        FileNotFoundError: If git is not found on PATH

    '''

    result = run_git(['push', remote_url, 'HEAD'], cwd=repo_root)
    return result.returncode == 0, result.stderr.strip()


def git_clone(remote_url: str, dest: Path) -> tuple[bool, str]:

    '''
    Clone a git repository into a destination directory, preserving history.

    Args:
        remote_url (str): Source git remote URL
        dest (Path): Destination directory for the clone

    Returns:
        tuple[bool, str]: (succeeded, stderr) where stderr carries git's
            error output when the clone fails

    Raises:
        FileNotFoundError: If git is not found on PATH

    '''

    result = run_git(['clone', remote_url, str(dest)])
    return result.returncode == 0, result.stderr.strip()


_FALLBACK_IDENTITY = ['-c', 'user.name=talos', '-c', 'user.email=talos@localhost']


def git_snapshot(repo_root: Path, message: str) -> tuple[bool, str]:

    '''
    Stage all changes and commit them, capturing the project's current state.

    Honors .gitignore, so ignored paths (e.g. results/dev/) are never staged.
    A clean working tree is a success with nothing committed. When the
    repository has no configured git user, commits under a talos identity so
    backups never fail for lack of git config.

    Args:
        repo_root (Path): Root directory of the git repository
        message (str): Commit message for the snapshot

    Returns:
        tuple[bool, str]: (succeeded, stderr) where stderr carries git's
            error output when staging or committing fails

    Raises:
        FileNotFoundError: If git is not found on PATH

    '''

    add = run_git(['add', '-A'], cwd=repo_root)
    if add.returncode != 0:
        return False, add.stderr.strip()

    if run_git(['diff', '--cached', '--quiet'], cwd=repo_root).returncode == 0:
        return True, ''

    commit_args = ['commit', '-m', message]
    if not _has_git_identity(repo_root):
        commit_args = [*_FALLBACK_IDENTITY, *commit_args]
    commit = run_git(commit_args, cwd=repo_root)
    return commit.returncode == 0, commit.stderr.strip()


def _has_git_identity(repo_root: Path) -> bool:

    name = run_git(['config', 'user.name'], cwd=repo_root)
    email = run_git(['config', 'user.email'], cwd=repo_root)
    return bool(name.stdout.strip()) and bool(email.stdout.strip())
