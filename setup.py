from setuptools import setup, find_packages

setup(name='pairwise-tomography',
      install_requires=['qiskit>=0.12', 'scipy', 'matplotlib', 'networkx'],
      version='0.0.2',
      packages=[package for package in find_packages()
                if package.startswith('pairwise_tomography')]
)
