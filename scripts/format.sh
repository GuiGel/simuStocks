#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place simustocks tests --exclude=__init__.py
black simustocks tests
isort simustocks tests
