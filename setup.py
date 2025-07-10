from setuptools import find_packages, setup

VERSION = "0.1.1"
DESCRIPTION = (
    "A framework for preprocessing email data, building, testing, and "
    "validating a zero-day phishing detection model using NLP and "
    "reconstruction techniques."
)

# Read long description from README.md
with open("README.md", "r", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

# Setting up
setup(
    name="zdpd_model",
    version=VERSION,
    author="Emilio Padron",
    author_email="emiliopm1997@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "transformers",
        "torch"
    ],
    keywords=[
        "python",
        "zero-day phishing",
        "phishing detection",
        "NLP",
        "email security",
        "reconstruction model",
        "cybersecurity",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    include_package_data=True
)
