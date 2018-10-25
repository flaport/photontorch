import photontorch
import setuptools

description = """Photonic simulation tools for highly parallel simulation and optimization
of photonic circuits in time and frequency domain."""

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="photontorch",
    version=photontorch.__version__,
    author="Floris Laporte",
    author_email="floris.laporte@ugent.be",
    description=description.replace("\n", " "),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.ugent.be/fplaport/photontorch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 2.7",
        "Development Status :: 3 - Alpha",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
