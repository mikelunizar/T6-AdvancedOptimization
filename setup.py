from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mnist_cnn_lightning",
    version="0.1.0",
    author="Your Name",
    author_email="mikel.martinez@unizar.es",
    description="PyTorch Lightning CNN for MNIST classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mikelunizar/T6-AdvancedOptimization",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
