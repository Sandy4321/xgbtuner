"""
xgbtuner setup script, derived from
https://github.com/pypa/sampleproject/blob/master/setup.py
"""
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='xgbtuner',
    version='1.0.0',
    description='Tune the hyper parameters of the xgboost algorithm',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/zkurtz/xgbtuner',

    # Author details
    author='Zach Kurtz',
    author_email='zkurtz@gmail.com',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 2 - preAlpha',

        # Indicate who your project is intended for
        'Intended Audience :: Data scientists',
        'Topic :: Machine learning :: hyperparameter optimization',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.5',
    ],

    # What does your project relate to?
    keywords='xgboost hyperparameter optimization',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['pandas', 'xgboost'],
    
    package_data={
        'xgbtuner': ['/data/fake_multinomial_data.csv'],
    },
)
