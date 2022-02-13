#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.readlines()

test_requirements = ['pytest>=3', ]

setup(
    author="Gianluca Pacchiella",
    author_email='gp@kln2.org',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Library to handle capture and analysis of power traces",
    entry_points={
        'console_scripts': [
            'power_analysis=power_analysis.cli:main',
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme,
    include_package_data=True,
    keywords='power_analysis',
    name='power_analysis',
    packages=find_packages(include=['power_analysis', 'power_analysis.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/gipi/power-analysis',
    version='0.1.0',
    zip_safe=False,
)
