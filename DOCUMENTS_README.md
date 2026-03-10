# 

##  PDF  DOC 

###  1: 

 PDF  DOC  `data/documents/` ：

```bash
# （）
mkdir -p data/documents

# 
cp your_file1.pdf data/documents/
cp your_file2.pdf data/documents/
cp your_file.doc data/documents/
```

###  2: 

：

```bash
pip install -r requirements.txt
```

###  3: 

：

```bash
python load_documents.py
```

：
1.  `data/documents/`  PDF、DOC、DOCX 
2. （chunks）
3.  TF-IDF 
4. 

###  4: 

，：
- 
- 
- 

 `data/vector_store/` 。

## 

- ✅ PDF (`.pdf`)
- ✅ DOCX (`.docx`)
- ✅ DOC (`.doc`) -  Word 

## 

 `load_documents.py` ：

- `chunk_size`: （ 500 ）
- `chunk_overlap`: （ 50 ）

## 

```python
from app.tools.document_loader import DocumentLoader
from app.tools.vector_store import SimpleVectorStore

# 
loader = DocumentLoader()
chunks = loader.load_file("data/documents/your_file.pdf")

# 
chunks = loader.load_directory("data/documents")

# 
vector_store = SimpleVectorStore()
vector_store.add_chunks(chunks)

# 
results = vector_store.search("NIH Stage Model", top_k=5)
```

## 

， `load_documents.py`，：
- 
- 
- 

## 

1. ****: 
2. ****: 
3. ****:  TF-IDF ，（ Pinecone、Weaviate）
4. ****:  `data/vector_store/`，

## 

### ：

 `ModuleNotFoundError`，：

```bash
pip install PyPDF2 pdfplumber python-docx docx2txt numpy scikit-learn
```

### ：

- 
- 
- ，

### ：

- （）
- 
- 

## 

，RAG 。VectorTool 。
