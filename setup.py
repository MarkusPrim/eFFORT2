from setuptools import setup, find_packages
from pkg_resources import resource_filename
from pathlib import Path


with open(resource_filename("EFFORT", "version.txt"), "r") as vf:
    version = vf.read().strip()

with (Path(__file__).parent / 'readme.md').open() as readme_file:
    readme = readme_file.read()

setup(
    name='EFFORT2',
    packages=find_packages(),
    url=None # 'https://github.com/b2-hive/eFFORT', old version
    author='Markus Tobias Prim',
    author_email='markus.prim@cern.ch',
    description='A tool for convenient reweighting between different form '
                'factors of semileptonic B decays.',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'tabulate',
        'uncertainties',
        'numdifftools',
        'pandas',
    ],
    include_package_data=True,
    long_description=readme,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License "
    ],
    license='MIT',
)
