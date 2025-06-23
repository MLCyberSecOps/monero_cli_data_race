#!/usr/bin/env python3
"""ThreadGuard: Advanced Static Analysis for Concurrency Bug Detection."""

import os

from setuptools import find_packages, setup

# Read the contents of README.md for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="threadguard",
    version="2.0.0",
    author="Pradeep Kumar",
    author_email="pradeep@example.com",
    description="Advanced static analysis tool for detecting concurrency issues in C++ codebases",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MLCyberSecOps/monero_cli_data_race",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    package_data={
        "": ["*.txt", "*.md", "*.rst", "*.yaml", "*.ini"],
    },
    entry_points={
        "console_scripts": [
            "threadguard=threadguard_enhanced:main",
        ],
    },
    python_requires=">=3.7",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Software Development :: Testing",
        "Topic :: Security",
    ],
    keywords="static-analysis concurrency deadlock-detection data-races thread-safety",
    project_urls={
        "Bug Reports": "https://github.com/MLCyberSecOps/monero_cli_data_race/issues",
        "Source": "https://github.com/MLCyberSecOps/monero_cli_data_race",
    },
)
