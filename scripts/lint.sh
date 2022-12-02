
#!/usr/bin/env bash

set -e
set -x

mypy typer
black simustocks tests --check
isort simustocks tests --check-only
