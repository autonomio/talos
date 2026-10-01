from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd


def read_from_file(_self: Any, file_path: str) -> pd.DataFrame:

    '''
    Create cleaned experiment log DataFrame from file.

    Args:
        file_path (str): Path to experiment log CSV file

    Returns:
        pd.DataFrame: Cleaned log data with whitespace-trimmed object columns
    '''

    with Path(file_path).open() as f:

        header_line = f.readline()

        if header_line:
            header = header_line.strip()
            lines = [
                header_line,
                *[
                    line
                    for line in f
                    if line.strip() != header
                ],
            ]
        else:
            lines = []

    data = pd.read_csv(StringIO(''.join(lines)))

    for col in data.columns:
        if pd.api.types.is_object_dtype(data[col]) or pd.api.types.is_string_dtype(data[col]):

            data[col] = data[col].str.strip()

    return data
