"""A setuptools based setup module.
Taken from:
https://github.com/pypa/sampleproject/blob/master/setup.py

See also:
http://python-packaging-user-guide.readthedocs.org/en/latest/distributing/

Additional links:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""
from __future__ import print_function

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# http://stackoverflow.com/questions/2058802/how-can-i-get-the-version-defined-in-setup-py-setuptools-in-my-package
version = ""
with open(path.join(here, 'version')) as version_file:
    version = version_file.read().strip()

setup(
    name='skl-eval',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=version,

    description='A Python program to run iteratively sklearn evaluations and store them in a HDF file',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/rizac/skl-eval',

    # Author details
    author='riccardo zaccarelli',
    author_email='rizac@gfz-potsdam.de',  # FIXME: what to provide?

    # Choose your license
    license='GPL3',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ],

    # What does your project relate to?
    keywords='machine learning evalation',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'htmlcov']),

    python_requires='>=3.6.9',

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'PyYAML>=5.4.1',
        'pandas>=1.2.3',
        'scikit-learn>=0.24.2',
        'tables>=3.6.1',
        'click>=7.1.2'
    ],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        # use latest versions. Without boundaries
        'test': [
            'pytest>=6.2.2',
            'pytest-cov>=2.11.1'
        ],
        'jupyter': [
            'jupyter>=1.0.0',
            'jupyterlab>=3.0.16'
        ]
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here. If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well. The value must be a mapping
    # from package name to a list of relative path names that should be copied
    # into the package. The paths are interpreted as relative to the directory
    # containing the package
    package_data={
       'tests': ['data'],
    },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    #
    # data_files=[('my_data', ['data/data_file'])],

    # For info see https://python-packaging.readthedocs.io/en/latest/non-code-files.html
    # include_package_data=True,
    # zip_safe=False,

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'skl-eval=skleval.evaluate:run',
        ],
    },
)

# print(str(find_packages(exclude=['contrib', 'docs', 'tests', 'htmlcov'])))
