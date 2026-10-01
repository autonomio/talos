"""Lazy model adapters shared by Scan, SFDs and the command API."""
from .adapters import backend_for, normalise_result, reference, resolve

__all__ = ['backend_for', 'normalise_result', 'reference', 'resolve']
