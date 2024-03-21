#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place "merge_tokenizers" "scripts" --exclude=__init__.py
isort "merge_tokenizers" "scripts"
black "merge_tokenizers" "scripts" -l 80
