from setuptools import setup, find_packages

setup(
    name='install_demo',
    version='0.1.2',
    author='Roland Potthast',
    author_email='Roland.Potthast@dwd.de',
    description='A simple Python package with greeting functions',
    packages=find_packages(),
    python_requires='>=3.6',
)