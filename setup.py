from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

setup(
    name='linear_regression',
    version='0.1',
    packages=find_packages(where='linear_regression'),
    install_requires=[
        "numpy",
        "torch",
        "scikit-learn",
        "matplotlib"
    ],
    package_dir={'': 'linear_regression'},
    py_modules=[splitext(basename(path))[0] for path in glob('linear_regression/*.py')],
    description='Uses Torch and Scikit-learn to demonstrate how to perform linear regression',
    author='Alberto J. Garcia',
    zip_safe=False
)
