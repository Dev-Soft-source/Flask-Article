#!/bin/bash
# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

# Script to test the grammar checking functionality in Script 2
# This script runs the test_grammar_fix.py script

# Navigate to the script2 directory
cd "$(dirname "$0")"

# Ensure required packages are installed
pip install tiktoken requests tenacity openai python-dotenv

# Run the test script
echo "Running grammar check test..."
python test_grammar_fix.py

# Exit with the status of the test
exit $?
