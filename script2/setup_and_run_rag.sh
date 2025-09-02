#!/bin/bash

# Display script starting message
echo "Starting RAG-enabled article generator setup..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Make sure keyword.txt exists and has content
if [ ! -f "input.csv" ]; then
    echo "Creating input.csv file..."
fi

# Display the keyword being used
echo "Using keyword from input.csv:"
cat input.csv

# Run the script
echo "Running article generator with RAG capabilities..."
python3 main.py --input input.csv

# Deactivate virtual environment
deactivate

echo "Article generation complete. Check the generated_articles directory for results." 