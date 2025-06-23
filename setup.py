"""
Setup script for Enhanced MMaDA package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    requirements = (this_directory / "requirements.txt").read_text().strip().split('\n')

setup(
    name="enhanced-mmada",
    version="1.0.0",
    author="Enhanced MMaDA Team",
    author_email="team@enhanced-mmada.ai",
    description="Enhanced Multimodal Attention and Domain Adaptation Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
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
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=1.0",
            "pre-commit>=2.20",
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.18",
        ],
        "viz": [
            "matplotlib>=3.5",
            "seaborn>=0.11",
            "plotly>=5.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "enhanced-mmada=enhanced_mmada.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "enhanced_mmada": ["data/*", "configs/*"],
    },
    zip_safe=False,
) 