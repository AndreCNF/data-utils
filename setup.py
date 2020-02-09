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
        'comet-ml==3.*,>=3.0.2', 'dask==2.*,>=2.5.0',
        'distributed==2.*,>=2.5.0', 'everett==1.*,>=1.0.2',
        'fsspec==0.*,>=0.6.0', 'modin==0.*,>=0.7.0', 'pandas==0.*,>=0.25.1',
        'pdoc3==0.*,>=0.7.2', 'pillow==7.*,>=7.0.0', 'portray==1.*,>=1.3.0',
        'ray==0.8.0', 'sklearn==0.*,>=0.0.0', 'sphinx==2.*,>=2.2.0',
        'torch==1.*,>=1.2.0', 'tqdm==4.*,>=4.38.0'
    ],
    extras_require={'dev': ['autopep8==1.*,>=1.4.0', 'pytest==3.*,>=3.0.0']},
)
