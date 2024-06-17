# -*- coding: utf-8 -*-
from setuptools import setup

packages = ["trainer"]

package_data = {"": ["*"]}

install_requires = [
    "Pillow>=8.4.0,<9.0.0",
    "gcsfs>=2021.10.1,<2022.0.0",
    "matplotlib>=3.4.3,<4.0.0",
    "pytorch_lightning>=1.4.9,<2.0.0",
    "tensorboard>=2.7.0,<3.0.0",
    "torch>=1.10.0,<2.0.0",
]

setup_kwargs = {
    "name": "trainer",
    "version": "0.1.0",
    "description": "",
    "long_description": None,
    "author": "Craig ",
    "author_email": "ctr26@ebi.ac.uk",
    "maintainer": None,
    "maintainer_email": None,
    "url": None,
    "packages": packages,
    "package_data": package_data,
    "install_requires": install_requires,
    "python_requires": ">=3.7,<3.8",
}


setup(**setup_kwargs)
