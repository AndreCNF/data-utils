from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='data-utils',
    version='0.0.1',
    author='André Ferreira',
    author_email='andrecnf@gmail.com',
    description='A set of generic, useful data science and machine learning methods.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/AndreCNF/data-utils',
    setup_requires=['setuptools_scm'],
    use_scm_version=True,
    packages=find_packages(),
    include_package_data=True,
)
