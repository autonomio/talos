#! /usr/bin/env python
#
# Copyright (C) 2018 Mikko Kotila

DESCRIPTION = "Hyperio Hyperparameter Scanner for Keras"
LONG_DESCRIPTION = """\
Hyperio provides a hyperparameter scanning solution that
allows using any Keras model as they are, with the simple
change that instead of calling the parameter (e.g. epochs=25),
you call it from a dictionary with an identical label (e.g. params['epochs']).

Really, not kidding.
"""

DISTNAME = 'hyperio'
MAINTAINER = 'Mikko Kotila'
MAINTAINER_EMAIL = 'mailme@mikkokotila.com'
URL = 'http://autonom.io'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/autonomio/hyperio/'
VERSION = '0.1.6'

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
        import tensorflow
    except ImportError:
        install_requires.append('tensorflow')
    try:
        import keras
    except ImportError:
        install_requires.append('keras')
    try:
        import astetik
    except ImportError:
        install_requires.append('astetik')


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
        packages=['hyperio',
                  'hyperio.data',
                  'hyperio.utils'],

        classifiers=[
                     'Intended Audience :: Science/Research',
                     'Programming Language :: Python :: 2.7',
                     'License :: OSI Approved :: MIT License',
                     'Topic :: Scientific/Engineering :: Human Machine Interfaces',
                     'Topic :: Scientific/Engineering :: Artificial Intelligence',
                     'Topic :: Scientific/Engineering :: Mathematics',
                     'Operating System :: POSIX',
                     'Operating System :: Unix',
                     'Operating System :: MacOS'],
)
