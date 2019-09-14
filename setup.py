from setuptools import setup, find_packages

setup(name='pairwise-tomography',
      install_requires=['qiskit', 'scipy', 'matplotlib', 'networkx'], # And any other dependencies foo needs
      version='0.0.1',
      packages=[package for package in find_packages()
                if package.startswith('pairwise_tomography')]
)
