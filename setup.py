"""
Setup script for SerenaNet.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="serenanet",
    version="1.0.0",
    author="SerenaNet Team",
    author_email="serenanet@outlook.com",
    description="A state-of-the-art speech recognition model with multi-modal architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CelsiaSolaraStarflare/SerenaArch2026",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
            "pre-commit>=3.3.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "notebook>=6.5.0",
            "ipywidgets>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "serena-train=src.training.trainer:main",
            "serena-eval=src.evaluation.metrics:main",
            "serena-preprocess=src.data.preprocessing:main",
        ],
    },
    include_package_data=True,
    package_data={
        "serenanet": ["configs/*.yaml"],
    },
)
