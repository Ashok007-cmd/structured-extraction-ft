#!/usr/bin/env bash
# Run the test suite with correct PYTHONPATH set and using the virtual environment

# Get the directory of the script and resolve to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Run tests
if [ -f "$PROJECT_ROOT/.venv/bin/pytest" ]; then
    echo "Running tests using virtual environment..."
    if [ $# -eq 0 ]; then
        "$PROJECT_ROOT/.venv/bin/pytest" "$PROJECT_ROOT/tests"
    else
        "$PROJECT_ROOT/.venv/bin/pytest" "$@"
    fi
else
    echo "Virtual environment pytest not found. Running with system pytest..."
    if [ $# -eq 0 ]; then
        pytest "$PROJECT_ROOT/tests"
    else
        pytest "$@"
    fi
fi
