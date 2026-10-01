"""Compatibility entrypoint for the maintained pytest acceptance suite."""
import sys
import pytest

if __name__ == '__main__':
    sys.exit(pytest.main(['tests', '-q', *sys.argv[1:]]))
