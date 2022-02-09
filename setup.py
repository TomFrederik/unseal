from setuptools import setup, find_packages

setup(
    name='Unseal',
    version='0.0.1',
    packages=find_packages(include=['unseal', 'unseal.*']),
    python_requires='>3.9.0',
    install_requires=[
        'pytorch>=1.10.0',
        'einops>=0.3.2',
        'numpy>=1.21.2',
        'transformers>=4.16.0',
        'tqdm',
        'matplotlib',
    ],
)
