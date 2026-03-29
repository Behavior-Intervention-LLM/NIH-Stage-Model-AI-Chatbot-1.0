#!/bin/bash
# ：

echo "=========================================="
echo "NIH Stage Model AI Chatbot"
echo "=========================================="

# 
if [ ! -d "venv" ]; then
    echo "..."
    python3 -m venv venv
fi

# 
source venv/bin/activate

# 
echo "..."
pip install -q -r requirements.txt

# 
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✓ "
else
    echo "..."
    # 
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
    BACKEND_PID=$!
    echo " PID: $BACKEND_PID"
    
    # 
    echo "..."
    sleep 3
    
    # 
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ "
    else
        echo "✗ ， backend.log"
        exit 1
    fi
fi

echo ""
# ，
FRONTEND_PORT=8501
if lsof -Pi :$FRONTEND_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️   $FRONTEND_PORT ，..."
    for port in 8502 8503 8504 8505; do
        if ! lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
            FRONTEND_PORT=$port
            echo "✓ : $FRONTEND_PORT"
            break
        fi
    done
    if [ $FRONTEND_PORT -eq 8501 ]; then
        echo "✗ ，"
        exit 1
    fi
else
    echo "✓  $FRONTEND_PORT "
fi

echo ""
echo "..."
echo ""
echo ""
echo ": http://localhost:$FRONTEND_PORT"
echo " API: http://localhost:8000"
echo ""
echo " Ctrl+C "
echo "=========================================="

# 
streamlit run frontend_streamlit.py --server.port $FRONTEND_PORT --server.headless true
