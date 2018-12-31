#! /usr/bin/env python
#
# Copyright (C) 2018 Mikko Kotila

DESCRIPTION = "Talos Hyperparameter Scanner for Keras"
LONG_DESCRIPTION = """\
Talos provides a hyperparameter scanning solution that
allows using any Keras model as they are, with the simple
change that instead of calling the parameter (e.g. epochs=25),
you call it from a dictionary with an identical label (e.g. params['epochs']).
"""

DISTNAME = 'talos'
MAINTAINER = 'Mikko Kotila'
MAINTAINER_EMAIL = 'mailme@mikkokotila.com'
URL = 'http://autonom.io'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/autonomio/talos/'
VERSION = '0.4.6'

try:
    from setuptools import setup
    _has_setuptools = True
except ImportError:
    from distutils.core import setup


def check_dependencies():

    install_requires = []

    try:
        import numpy
    except ImportError:
        install_requires.append('numpy')
    try:
        import pandas
    except ImportError:
        install_requires.append('pandas')
    try:
        import keras
    except ImportError:
        install_requires.append('keras')
    try:
        import astetik
    except ImportError:
        install_requires.append('astetik')
    try:
        import sklearn
    except ImportError:
        install_requires.append('sklearn')
    try:
        import tqdm
    except ImportError:
        install_requires.append('tqdm')
    try:
        import chances
    except ImportError:
        install_requires.append('chances')
    try:
        import kerasplotlib
    except ImportError:
        install_requires.append('kerasplotlib')
    try:
        import wrangle
    except ImportError:
        install_requires.append('wrangle')
    try:
        import requests
    except ImportError:
        install_requires.append('requests')

    return install_requires


if __name__ == "__main__":

    install_requires = check_dependencies()

    setup(name=DISTNAME,
          author=MAINTAINER,
          author_email=MAINTAINER_EMAIL,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          install_requires=install_requires,
          packages=['talos',
                    'talos.scan',
                    'talos.examples',
                    'talos.utils',
                    'talos.model',
                    'talos.parameters',
                    'talos.reducers',
                    'talos.metrics',
                    'talos.commands'],

          classifiers=[
                 'Intended Audience :: Science/Research',
                 'Programming Language :: Python :: 3.6',
                 'License :: OSI Approved :: MIT License',
                 'Topic :: Scientific/Engineering :: Human Machine Interfaces',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Topic :: Scientific/Engineering :: Mathematics',
                 'Operating System :: POSIX',
                 'Operating System :: Unix',
                 'Operating System :: MacOS'],)
