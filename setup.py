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
        "merge_tokenizers.aligners.dtw_c.dtw",
        ["merge_tokenizers/aligners/dtw_c/dtw.c"],
        include_dirs=["merge_tokenizers/aligners/dtw_c"],
    ),
]


setup(
    version=VERSION["VERSION"],
    name="merge-tokenizers",
    description="Package to merge tokens from different tokenizers.",
    author="Symanto Research GmbH",
    author_email="jose.gonzalez@symanto.com",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRES,
    include_package_data=True,
    ext_modules=extensions,
    python_requires=">=3.8.0",
    zip_safe=False,
)
