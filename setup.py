#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup
from pulp.__version__ import __VERSION__

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='ProSper',
      version=__VERSION__,
      description='A Python Library for Probabilistic Sparse Coding with Learnable Priors and Different Superposition Assumptions',
      url='http://bitbucket.org/mlold/pylib',
      author='Jorg Bornschein, Abdul-Saboor Sheikh, Zhenwen Dai, Georgios Exarchakis, Marc Henniges, Julian Eggert, Jorg Lucke',
      author_email='bornj@iro.umontreal.ca',
      license='AFL3',
      packages=['pulp',
                'pulp.em',
                'pulp.utils',
                'pulp.em.camodels',
                'pulp.em.mixturemodels'],
      package_dir={'pulp': 'pulp'},
      long_description=read('README.md'),
      install_requires=['numpy>=1.7', 'scipy>=0.12','tables','nose>= 1.2','mpi4py'],
      classifiers=['License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 2.7',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence'],
)