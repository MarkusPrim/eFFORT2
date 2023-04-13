from setuptools import setup, find_packages
from pkg_resources import resource_filename
from pathlib import Path


with (Path(__file__).parent / 'README.md').open() as readme_file:
    readme = readme_file.read()

setup(
    name='EFFORT2',
    packages=find_packages(),
    url="",
    author='Markus Tobias Prim',
    author_email='markus.prim@cern.ch',
    description='''
A tool for convenient reweighting between different form
factors of semileptonic B decays, and fitting measured
spectra.
''',
    install_requires=[
        'numpy',
        'scipy',
        'numba',
        'setuptools',
        'uncertainties',
    ],
    extras_require={
        "examples":  ['matplotlib', 'jupyterlab', 'gvar'],
    },
    long_description=readme,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License "
    ],
    license='MIT',
)
