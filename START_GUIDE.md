# 

## （）

###  1: （ + ）

```bash
./start.sh
```

：
1. （ 8000）
2. （ 8501）
3. 

###  2: 

** 1 - ：**
```bash
./start_backend.sh
# 
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

** 2 - ：**
```bash
./start_frontend.sh
# 
streamlit run frontend_streamlit.py
```

## 

：

- ****: http://localhost:8501
- ** API**: http://localhost:8000
- **API **: http://localhost:8000/docs
- ****: http://localhost:8000/health

## 

1. **（）**
   ```bash
   #  PDF/DOC  data/documents/
   python load_documents.py
   ```

2. ****
   ```bash
   ./start.sh
   ```

3. ****
   -  http://localhost:8501
   - ！

## 

### 

- ✅ ： ChatGPT 
- ✅ ：、
- ✅ ： Agent 
- ✅ ： RAG 
- ✅ ：

### 

- ****：、
- ****：/
- ****：
- ****：

### 

，：
- （route_mode）
-  Agents
- Stage 
- 

## 

### ：

** 8501 ：**
-  8502、8503 
- ：
```bash
# 
lsof -i :8501

# （ PID  ID）
kill <PID>

# 
./kill_ports.sh
```

** 8000 ：**
```bash
# 
lsof -i :8000

# 
kill <PID>

# 
./kill_ports.sh
```

### ：

**：**
```bash
lsof -i :8000
# ， app/main.py 
```

**：**
```bash
pip install -r requirements.txt
```

### ：

1. （ http://localhost:8000/health）
2.  `frontend_streamlit.py`  `BACKEND_URL` 
3. 

### ：

```bash
# 
ls data/documents/

# 
python load_documents.py
```

## Windows 

 `start.sh` ，：

**：**
```cmd
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**：**
```cmd
streamlit run frontend_streamlit.py
```

## 

，：

** 1（）：**
```bash
uvicorn app.main:app --reload
```

** 2（）：**
```bash
streamlit run frontend_streamlit.py
```

：
- 
- 
- 

## 

：

1. ****（ PM2、supervisor）
2. ****（ Nginx）
3. ** HTTPS**
4. ****（API keys ）

（ PM2）：
```bash
pm2 start "uvicorn app.main:app --host 0.0.0.0 --port 8000" --name backend
pm2 start "streamlit run frontend_streamlit.py --server.port 8501" --name frontend
```
