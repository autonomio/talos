#!/usr/bin/env python3
"""Resolve the protected comparison ref and print only its validated identity."""
from __future__ import annotations

import argparse

from _common import comparison_ref


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-ref', required=True)
    parser.add_argument('--head-ref', default='HEAD')
    args = parser.parse_args()
    print(comparison_ref(args.base_ref, args.head_ref))


if __name__ == '__main__':
    main()
