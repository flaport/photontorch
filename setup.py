import setuptools
import photontorch

try:
    with open("README.md", "r") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = photontorch.__doc__

setuptools.setup(
    name="photontorch",
    version=photontorch.__version__,
    author="Floris Laporte",
    author_email="floris.laporte@ugent.be",
    description=photontorch.__doc__.split("\n")[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/flaport/photontorch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 2.7",
        "Development Status :: 3 - Alpha",
        "Operating System :: OS Independent",
        "License :: Free for non-commercial use",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
