from typing import List, Dict
from setuptools import setup, find_packages, Extension

VERSION: Dict[str, str] = {}
with open("merge_tokenizers/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

INSTALL_REQUIRES: List[str] = [
    "levenshtein",
    "pydantic",
    "spacy-alignments",
    "scikit-learn",
    "fastdtw",
    "ukkonen",
    "numba",
]

EXTRAS_REQUIRES: Dict[str, List[str]] = {
    "dev": [
        "black",
        "flake8",
        "mypy",
        "types-requests",
        "pytest",
        "isort",
        "autoflake",
        "pre-commit",
    ]
}

extensions = [
    Extension(
        name="merge_tokenizers.aligners.dtw_c.dtw",
        sources=["merge_tokenizers/aligners/dtw_c/dtw.c"],
        language="c",
        include_dirs=["merge_tokenizers/aligners/dtw_c"],
    ),
    Extension(
        name="merge_tokenizers.aligners.greedy_coverage_c.greedy_coverage",
        sources=["merge_tokenizers/aligners/greedy_coverage_c/greedy_coverage.c"],
        language="c",
        include_dirs=["merge_tokenizers/aligners/greedy_coverage_c"],
    ),
]


setup(
    version=VERSION["VERSION"],
    name="merge-tokenizers",
    description="Package to merge tokens from different tokenizers.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Symanto Research GmbH",
    author_email="jose.gonzalez@symanto.com",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRES,
    include_package_data=True,
    ext_modules=extensions,
    zip_safe=False,
    python_requires=">=3.8.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
    ],
    license_files=[
        "LICENSE",
    ],
)
