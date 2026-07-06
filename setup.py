"""
setup.py — Package installer for the ``microex`` micro-expression toolkit.

Usage
-----
Install in development (editable) mode::

    pip install -e .

Build a wheel::

    python -m build
"""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent

# Read the long description from README if available.
readme_path = ROOT / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Parse requirements.txt
requirements_path = ROOT / "requirements.txt"
install_requires: list[str] = []
if requirements_path.exists():
    install_requires = [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="microex",
    version="0.1.0",
    description="Micro-Expression Detection System using OpenCV",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Micro-Expression Detection Team",
    license="MIT",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black",
            "mypy",
            "ruff",
        ],
    },
    entry_points={
        "console_scripts": [
            "microex=microex.__main__:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
