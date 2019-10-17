#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup
from prosper.__version__ import __VERSION__

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='ProSper',
      version=__VERSION__,
      description='A Python Library for Probabilistic Sparse Coding with Learnable Priors and Different Superposition Assumptions',
      url='http://github.com/ml-uol/prosper',
      author='Jorg Bornschein, Abdul-Saboor Sheikh, Zhenwen Dai, Georgios Exarchakis, Marc Henniges, Julian Eggert, Jorg Lucke',
      author_email='bornj@iro.umontreal.ca',
      license='AFL3',
      packages=['prosper',
                'prosper.em',
                'prosper.utils',
                'prosper.em.camodels',
                'prosper.em.mixturemodels'],
      package_dir={'prosper': 'prosper'},
      long_description=read('README.md'),
      install_requires=['numpy>=1.15', 'scipy>=1.3','tables','nose>= 1.2','mpi4py'],
      classifiers=['License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 2.7',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence'],
      command_options={
        'build_sphinx': {
            'project': ('setup.py', 'ProSper'),
            'version': ('setup.py', __VERSION__),
            'release': ('setup.py', __VERSION__),
            'source_dir': ('setup.py', 'docs')}}
)