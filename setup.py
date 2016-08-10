#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup
from pulp.__version__ import __VERSION__

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='pylib',
      version=__VERSION__,
      description='TBD',
      url='TBD',
      author='TBD',
      author_email='TBD',
      license='AFL3',
      packages=['pulp',
                'pulp.em',
                'pulp.utils',
                'pulp.em.camodels',
                'pulp.em.mixturemodels'],
      package_dir={'pulp': 'pulp'},
      long_description=read('README.md'),
      install_requires=['numpy>=1.7', 'scipy>=0.16'],
      classifiers=['License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 2.7',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence'],
)