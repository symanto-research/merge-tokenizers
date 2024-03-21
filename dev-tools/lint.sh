#!/usr/bin/env bash

set -e
set -x

mypy "merge_tokenizers"
flake8 "merge_tokenizers" --ignore=E501,W503,E203,E402
black "merge_tokenizers" --check -l 80
