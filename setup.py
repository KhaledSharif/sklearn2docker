from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='sklearn2docker',
    version='0.1',
    description='Convert your trained scikit-learn classifier to a Docker container with a pre-configured API.',
    long_description=long_description,
    url='https://github.com/KhaledSharif/sklearn2docker',
    author='Khaled Sharif',
    author_email='kldsrf@gmail.com',
    keywords='machine-learning docker deployment data-science',
    packages=find_packages(exclude=['contrib', 'docs', 'tests', 'notebooks', 'examples']),
    install_requires=['pandas', 'scipy', 'sklearn'],
)