# -*- coding: utf-8 -*-

# DO NOT EDIT THIS FILE!
# This file has been autogenerated by dephell <3
# https://github.com/dephell/dephell

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

readme = ''

setup(
    long_description=readme,
    name='data-utils',
    version='0.1.0',
    description='A set of generic, useful data science and machine learning methods.',
    python_requires='==3.*,>=3.6.0',
    project_urls={
        'homepage': 'https://github.com/andrecnf/data_utils',
        'repository': 'https://github.com/andrecnf/data_utils'
    },
    author='AndreCNF',
    author_email='andrecnf@gmail.com',
    license='MIT',
    packages=['data_utils'],
    package_data={},
    install_requires=[
        'dask==2.*,>=2.5.0', 'distributed==2.*,>=2.5.0', 'pandas==0.*,>=0.25.1',
        'torch==1.*,>=1.2.0'
    ],
    extras_require={'dev': ['autopep8==1.*,>=1.4.0', 'pytest==3.*,>=3.0.0']},
)
