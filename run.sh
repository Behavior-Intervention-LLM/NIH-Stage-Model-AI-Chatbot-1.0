#!/bin/bash
# 

# 
if [ ! -d "venv" ]; then
    echo "..."
    python3 -m venv venv
fi

# 
source venv/bin/activate

# 
echo "..."
pip install -r requirements.txt

# 
echo "..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
