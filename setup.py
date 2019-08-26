#! /usr/bin/env python
#
# Copyright (C) 2018 Mikko Kotila

DESCRIPTION = "Talos Hyperparameter Tuning for Keras"
LONG_DESCRIPTION = """\
Talos radically changes the ordinary Keras workflow by
fully automating hyperparameter tuning and model evaluation.
Talos exposes Keras functionality entirely and there is no new
syntax or templates to learn.
"""

DISTNAME = 'talos'
MAINTAINER = 'Mikko Kotila'
MAINTAINER_EMAIL = 'mailme@mikkokotila.com'
URL = 'http://autonom.io'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/autonomio/talos/'
VERSION = '0.6.3'

try:
    from setuptools import setup
    _has_setuptools = True
except ImportError:
    from distutils.core import setup

install_requires = ['wrangle',
                    'numpy',
                    'pandas',
                    'keras',
                    'astetik',
                    'sklearn',
                    'tqdm',
                    'chances',
                    'kerasplotlib',
                    'requests']


if __name__ == "__main__":

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
                    'talos.templates',
                    'talos.utils',
                    'talos.model',
                    'talos.parameters',
                    'talos.reducers',
                    'talos.metrics',
                    'talos.commands',
                    'talos.logging',
                    'talos.autom8'],

          classifiers=['Intended Audience :: Science/Research',
                       'Programming Language :: Python :: 2.7',
                       'Programming Language :: Python :: 3.5',
                       'Programming Language :: Python :: 3.6',
                       'License :: OSI Approved :: MIT License',
                       'Topic :: Scientific/Engineering :: Human Machine Interfaces',
                       'Topic :: Scientific/Engineering :: Artificial Intelligence',
                       'Topic :: Scientific/Engineering :: Mathematics',
                       'Operating System :: POSIX',
                       'Operating System :: Unix',
                       'Operating System :: MacOS',
                       'Operating System :: Microsoft :: Windows :: Windows 10'])
