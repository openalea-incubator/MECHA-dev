from setuptools import setup, find_packages

setup(
    name="mecha",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'networkx',
        'lxml',
        'geopandas',
        'pylab',
    ],
    entry_points={
        'console_scripts': [
            'mecha=mecha.main:mecha',
        ],
    },
)
