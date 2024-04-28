#!/bin/bash

# Execute setup script
python setup_prdcat_tabel.py

# Start the application
uvicorn app.main:app --host 0.0.0.0 --port 8000

