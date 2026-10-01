from dataclasses import dataclass


@dataclass
class YAMLError:

    '''A single validation error or warning with location and suggestion.'''

    message: str
    path: str = ''
    line: int | None = None
    column: int | None = None
    suggestion: str | None = None


class ValidationError(Exception):

    '''Raised when YAML validation fails; carries all collected errors.'''

    def __init__(self, errors: list['YAMLError']) -> None:

        self.errors = errors
        lines = '\n'.join(f'  {e.path}: {e.message}' for e in errors)
        super().__init__(f'YAML validation failed with {len(errors)} error(s):\n{lines}')


class ResolutionError(Exception):

    '''Raised when a namespaced path cannot be resolved to a Python object.'''

    def __init__(self, path: str, allowed_namespaces: list[str]) -> None:

        self.path = path
        self.allowed_namespaces = allowed_namespaces
        super().__init__(
            f"Cannot resolve '{path}'. Allowed namespaces: {', '.join(allowed_namespaces)}"
        )


class GitError(Exception):

    '''Raised when Git operations fail or the YAML file is not in a repository.'''

    def __init__(self, path: str, message: str) -> None:

        self.path = path
        self.message = message
        super().__init__(f"Git error for '{path}': {message}")
