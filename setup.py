from setuptools import setup, find_packages

setup(
    name='longform_uq',
    version='0.1.0',
    packages=find_packages(where='lib/python'),
    package_dir={'': 'lib/python'},
)
