#!/bin/bash
echo "Starting Roomify FastAPI app..."
python -m uvicorn app:app --host=0.0.0.0 --port=${PORT:-8000}
