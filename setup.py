from typing import List, Dict
from setuptools import setup, find_packages, Extension

VERSION: Dict[str, str] = {}
with open("merge_tokenizers/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

with open("requirements.txt", encoding="utf-8") as req_fp:
    install_requires = req_fp.readlines()

EXTRAS_REQUIRES: Dict[str, List[str]] = {}
with open("dev-requirements.txt", encoding="utf-8") as dev_req_fp:
    EXTRAS_REQUIRES["dev"] = dev_req_fp.readlines()

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
    install_requires=install_requires,
    extras_require=EXTRAS_REQUIRES,
    include_package_data=True,
    ext_modules=extensions,
    python_requires=">=3.8.0",
    zip_safe=False,
)
