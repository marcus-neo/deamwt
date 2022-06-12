#!/bin/sh

echo "Initiating PyLint" && \
pylint ${PWD} && \
echo "PyLint Passed, Initiating Flake8" && \
flake8 --ignore=E203,W503 ${PWD} && \
echo "Flake8 Passed, Initiating PyDocStyle" && \
pydocstyle \
--add-select=D203,D212,D205,D200 \
--add-ignore=D211 --match='(?!__init__).*\.py' ${PWD} && \
echo "PyDocStyle Passed." && \
echo "Lint Test Completed Successfully."