from setuptools import setup
import setuptools

setup(
    name="haven-ai",
    version="0.7.3",
    description="Manage large-scale experiments",
    url="https://github.com/haven-ai/haven-ai",
    maintainer="Issam Laradji",
    maintainer_email="issam.laradji@gmail.com",
    license="MIT",
    packages=setuptools.find_packages(),
    zip_safe=False,
    python_requires=">=3.5",
    install_requires=[
        "requests>=0.0",
        "tqdm>=0.0",
        "matplotlib>=0.0",
        "numpy>=0.0",
        "pandas>=0.0",
        "notebook >= 4.0",
    ],
),
