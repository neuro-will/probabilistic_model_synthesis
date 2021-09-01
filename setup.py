from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='probabilistic_model_synthesis',
    version='0.1.0',
    author='William Bishop',
    author_email='bishopw@janelia.hhmi.org',
    packages=['probabilistic_model_synthesis'],
    python_requires='>=3.7.0',
    description='Code for performing probabilistic model synthesis.',
    long_description=long_description,
    install_requires=["matplotlib", "numpy", "pot"]
)
