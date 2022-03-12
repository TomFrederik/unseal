from setuptools import setup, find_packages
import os, sys
sys.path.insert(0, os.path.abspath("."))
from unseal import __version__

version = __version__


setup(
    name='Unseal',
    version=version,
    packages=find_packages(exclude=[]),
    python_requires='>=3.6.0',
    install_requires=[
        'torch',
        'einops>=0.3.2',
        'numpy',
        'transformers',
        'tqdm',
        'matplotlib',
        'streamlit',
    ],
    # entry_points={
    #     'console_scripts': [
    #         '"unseal compare" = unseal.commands.interfaces.compare_two_inputs:main',
    #     ]
    # },
    description=(
        "Unseal "
        "Collection of infrastructure and tools for research in "
        "transformer interpretability."
    ),
    author="The Unseal Team",
    author_email="tlieberum@outlook.de",
    url="https://unseal.readthedocs.io",
    license="Apache License 2.0",
    keywords=[
        "pytorch",
        "tensor",
        "machine learning",
        "neural networks",
        "interpretability",
        "transformers",
    ],
)
