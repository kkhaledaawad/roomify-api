#!/bin/bash

# Check if uvicorn is installed
if ! python -m pip show uvicorn > /dev/null 2>&1; then
    echo "Installing uvicorn and fastapi..."
    python -m pip install uvicorn fastapi
fi

# Start the application
echo "Starting application..."
python -m uvicorn app:app --host 0.0.0.0 --port 8000
