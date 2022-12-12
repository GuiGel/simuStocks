
#!/usr/bin/env bash

set -e
set -x

mypy simustocks
black simustocks tests --check
isort simustocks tests --check-only
