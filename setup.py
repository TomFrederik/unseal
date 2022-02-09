from setuptools import setup, find_packages

version = "0.0.1"


setup(
    name='Unseal',
    version=version,
    packages=find_packages(include=['unseal', 'unseal.*']),
    python_requires='>=3.9.0',
    install_requires=[
        'torch>=1.10.1',
        'einops>=0.3.2',
        'numpy>=1.21.2',
        'transformers>=4.16.0',
        'tqdm',
        'matplotlib',
    ],
)