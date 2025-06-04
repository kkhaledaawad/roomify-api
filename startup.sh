#!/bin/bash

echo "Starting Roomify application..."

# Navigate to the app directory
cd /home/site/wwwroot

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install basic requirements
echo "Installing uvicorn and fastapi..."
pip install uvicorn fastapi

# Start the application
echo "Starting uvicorn server..."
python -m uvicorn app:app --host 0.0.0.0 --port 8000