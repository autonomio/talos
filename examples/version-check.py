"""Report installed versions without importing optional training frameworks."""
import sys
from importlib.metadata import PackageNotFoundError, version
import talos

print('Python %s' % sys.version.split()[0])
print('Talos %s' % talos.__version__)
for package in ('numpy', 'pandas', 'polars', 'scikit-learn', 'keras', 'tensorflow', 'torch'):
    try:
        installed = version(package)
    except PackageNotFoundError:
        installed = 'not installed'
    print('%s %s' % (package, installed))
