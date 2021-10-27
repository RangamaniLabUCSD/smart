import sys
from setuptools import setup, find_packages

with open("README.md", "r") as handle:
    long_description = handle.read()


setup(
    # Self-descriptive entries which should always be present
    name='fenics-stubs',
    author='Justin Laughlin',
    author_email='justinglaughlin@gmail.com',
    url='https://github.com/justinlaughlin/stubs',
    description='STUBS is a biophysical simulation library that provides a level of abstraction to models, making it easier for users to develop, share, and simulate their mathematical models.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    platforms=['Linux', 'Mac OS-X'],
    version='0.1.2',
    license='LGPLv3',

    # Which Python importable modules should be included when your package is installed
    # Handled automatically by setuptools. Use 'exclude' to prevent some specific
    # subpackage(s) from being added, if needed
    #packages=find_packages(),
    packages=['stubs'],

    # Optional include package data to ship with your package
    # Customize MANIFEST.in if the general case does not suit your needs
    # Comment out this line to prevent the files from being packaged with your software
    #include_package_data=True,

    # Additional entries you may want simply uncomment the lines you want and fill in the data
    install_requires=[
        'matplotlib',
        'numpy>=1.16.0',
        'pandas',
        'Pint',
        'scipy>=1.1.0',
        'sympy',
        'tabulate',
        'termcolor',
        ],              # Required packages, pulls from pip if needed; do not use for Conda deployment
    # python_requires=">=3.5",          # Python version restrictions

    # Manual control if final package is compressible or not, set False to prevent the .egg from being made
    # zip_safe=False,

)
