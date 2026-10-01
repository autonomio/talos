from typing import Any


def select(context: dict[str, Any]) -> list[int | str]:
    '''Return every available permutation id.'''

    return sorted(
        context['available_permutation_ids'],
        key=lambda value: (0, value) if isinstance(value, int) else (1, str(value)),
    )
