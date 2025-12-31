#!/bin/bash
set -e

echo "Starting application..."
echo "Current directory: $(pwd)"
echo "Contents: $(ls -la)"

# Navigate to the correct directory
cd /home/site/wwwroot/Backend/app/api || cd Backend/app/api

echo "Now in: $(pwd)"
echo "Files here: $(ls -la)"

# Start gunicorn
exec gunicorn --bind=0.0.0.0:8000 --timeout 600 --workers 2 --access-logfile '-' --error-logfile '-' main:app