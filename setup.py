from setuptools import setup, Extension
import setuptools
import os
import sys

# get __version__, __author__, and __email__
exec(open("./curvestat/metadata.py").read())

setup(
    name='curvestat',
    version=__version__,
    author=__author__,
    author_email=__email__,
    url='https://github.com/jonassjuul/curvestat',
    license=__license__,
    description="Get and plot descriptive statistics for ensembles of curves.",
    long_description='',
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
    	'numpy>=1.18.4',
    	'matplotlib>=3.2.1',
    	'scipy>=1.4.1',
    ],
    tests_require=[],
    setup_requires=['pytest-runner'],
    classifiers=['License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7'
                 ],
    project_urls={
        'Contributing Statement': 'https://github.com/jonassjuul/curvestat/blob/master/CONTRIBUTING.md',
        'Bug Reports': 'https://github.com/jonassjuul/curvestat/issues',
        'Source': 'https://github.com/jonassjuul/curvestat/',
        #'PyPI': 'https://pypi.org/project/curvestat/',
    },
    include_package_data=True,
    zip_safe=False,
)
