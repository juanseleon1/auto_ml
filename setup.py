#!/usr/bin/env python

from setuptools import setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Auto ML",
    version="1.0",
    description="Auto ML task for Analitica II",
    license="MIT",
    author="Monica Perez, Santiago, , Juan León",
    packages=["automl", "eda", "performance", "utils", "training", "transformation"],
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "generate_eda=eda.entry_points:do_eda",
            "start_auto_ml=automl.entry_points:start_auto_ml",
        ]
    },
)
