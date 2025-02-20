from setuptools import setup, find_packages

setup(
    name='bayes_evals',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "pandas",
    ],
)