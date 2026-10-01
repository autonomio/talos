import json
import os
from pathlib import Path

from .serialization import decode, encode


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(encode(value), handle, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_json(path):
    with Path(path).open(encoding='utf-8') as handle:
        return decode(json.load(handle))


def append_round(path, record):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(encode(record), sort_keys=True, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def read_rounds(path, count=None):
    records = []
    if not Path(path).exists():
        return records
    with Path(path).open(encoding='utf-8') as handle:
        for number, line in enumerate(handle):
            if count is not None and number >= count:
                break
            try:
                records.append(decode(json.loads(line)))
            except json.JSONDecodeError as error:
                raise ValueError(f'Incomplete trial record {number} in {path}') from error
    return records


def truncate_rounds(path, count):
    records = read_rounds(path, count)
    temporary = Path(path).with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(encode(record), sort_keys=True, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
