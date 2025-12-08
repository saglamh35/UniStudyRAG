# 🎓 UniStudyRAG: Local, Multimodal AI Study Assistant

<div align="center">

**Privacy-first RAG system running on local hardware (RTX 2070 optimized). Combines Gemma 3 (4b) for reasoning and Llama 3.2 Vision for image analysis.**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://www.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 🚀 Key Features

### 🏗️ Layered Architecture
Modern, modular design with separation of concerns:
- **Configuration Management** (`config.py`) - Centralized settings with `.env` support
- **Ingestion Layer** (`modules/ingestion.py`) - PDF processing, Vision integration, and chunking
- **Vector Store Layer** (`modules/vectorstore.py`) - ChromaDB management and retrieval
- **LLM Engine** (`modules/llm_engine.py`) - Language model orchestration
- **Utilities** (`utils.py`) - Centralized logging

### 🧠 Hybrid Architecture
Vision + Text pipeline that processes both textual content and visual elements (diagrams, charts, schematics) from PDF documents.

### 🔒 100% Local & Private
Runs completely offline using Ollama. Your documents never leave your machine. Perfect for sensitive academic materials and confidential research papers.

### 👁️ Multimodal RAG with Aggressive OCR
Advanced vision processing that:
- **Extracts ALL text** including headers, titles, university names, course codes
- **Analyzes diagrams** with detailed node, arrow, and connection descriptions
- **Transcribes formulas** and mathematical expressions
- Uses **two-phase analysis** (OCR + Visual Description) for maximum accuracy

### ⚡ Smart Retrieval
Uses **MMR (Maximal Marginal Relevance)** algorithm to reduce retrieval bias. Ensures diverse context retrieval from different parts of documents, preventing the system from focusing only on similar chunks.

### 💾 Smart Caching System
Intelligent caching mechanism that:
- Caches processed PDFs using MD5 hash
- Skips Vision processing on cache hits (dramatically faster)
- Automatically invalidates when PDFs change
- Reduces processing time for repeated document uploads

### 🗣️ Elite Multilingual Chain-of-Thought
Advanced prompting strategy that:
- Handles German/English documents with Turkish (or any language) answers
- Distinguishes between document owners and references (critical for CVs and academic papers)
- Maintains technical terminology in original language while translating explanations
- Direct, concise answers without unnecessary processing steps
- Prioritizes OCR-extracted content for accurate document understanding

### 💬 Streaming Responses
ChatGPT-style streaming interface with real-time word-by-word response generation for a smooth user experience.

### 🪟 Windows Optimized
Special handling for Windows file locking issues with ChromaDB, ensuring smooth operation on Windows systems.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit (Modern web interface)
- **Backend:** Python 3.12+
- **LLM Framework:** LangChain
- **Models:** 
  - **Gemma 3 (4b)** - Text generation and reasoning
  - **Llama 3.2 Vision** - Image analysis and visual content understanding
  - **Nomic Embed Text** - Text embeddings for vector search
- **Vector Database:** ChromaDB
- **PDF Processing:** PyPDF, pdf2image
- **Runtime:** Ollama (Local inference)

---

## 📋 Prerequisites

1. **Python 3.12+** installed
2. **Ollama** installed and running ([Download](https://ollama.ai/))
3. **Poppler** (for PDF to image conversion):
   - **Windows:** Download from [Poppler for Windows](http://blog.alivate.com.au/poppler-windows/) and add to PATH
   - **macOS:** `brew install poppler`
   - **Linux:** `sudo apt-get install poppler-utils` (Ubuntu/Debian) or `sudo yum install poppler-utils` (RHEL/CentOS)

---

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/saglamh35/UniStudyRAG.git
cd UniStudyRAG
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment (Optional)

Create a `.env` file in the project root to customize settings:

```env
# Model Ayarları
LLM_MODEL_NAME=gemma3:4b
VISION_MODEL_NAME=llama3.2-vision
EMBED_MODEL_NAME=nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434

# RAG Ayarları
CHUNK_SIZE=600
CHUNK_OVERLAP=150
RETRIEVAL_K=8

# Vision Ayarı
ENABLE_VISION=true

# Path Ayarları (Optional)
CACHE_DIR=cache_data
CHROMA_DB_DIR=chroma_db
DATA_DIR=data
```

**Note:** If `.env` is not provided, default values from `config.py` will be used.

### 5. Install Ollama Models

```bash
# Pull the required models
ollama pull gemma3:4b
ollama pull llama3.2-vision
ollama pull nomic-embed-text
```

**Note:** Model downloads may take time depending on your internet connection. Gemma 3 (4b) is ~2.4GB, Llama 3.2 Vision is ~4.7GB, and Nomic Embed Text is ~274MB.

---

## 🎯 Usage

### Start the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Using the Application

1. **Upload PDFs:** Use the sidebar to upload one or multiple PDF files
2. **Wait for Processing:** The system will:
   - Extract text from PDFs
   - Convert pages to images
   - Analyze images with Llama 3.2 Vision
   - Create vector embeddings
   - Build the searchable database
3. **Ask Questions:** Type your questions in any language (Turkish, German, English, etc.)
4. **Get Answers:** Receive streaming responses with source citations

### Example Queries

- "Bu CV'deki kişinin eğitim geçmişi nedir?" (Turkish)
- "What are the main topics covered in this document?" (English)
- "Welche Technologien werden in diesem Dokument erwähnt?" (German)

---

## 📁 Project Structure

```
UniStudyRAG/
├── app.py                 # Streamlit web interface
├── rag_engine.py         # Core RAG manager (orchestrates modules)
├── main.py               # CLI version (alternative interface)
├── config.py             # Configuration management (.env support)
├── utils.py              # Centralized logging utilities
├── modules/              # Layered architecture modules
│   ├── __init__.py
│   ├── ingestion.py      # PDF loading, Vision, Chunking
│   ├── vectorstore.py    # ChromaDB, Embeddings, Retriever
│   └── llm_engine.py     # LLM orchestration, System Prompt
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (create from template)
├── .gitignore           # Git ignore rules
├── README.md            # This file
├── data/                # PDF storage (gitignored)
├── chroma_db/           # Vector database (gitignored)
└── cache_data/           # Smart cache storage (gitignored)
```

---

## 🔍 How It Works

### 1. Document Processing
- PDFs are loaded and split into manageable chunks (600 chars with 150 overlap)
- **Smart Caching:** MD5 hash check - if PDF was processed before, loads from cache instantly
- Each page is converted to an image
- **Aggressive OCR:** Vision model extracts ALL text (headers, titles, university names, course codes)
- **Two-Phase Analysis:** OCR extraction + Visual description for maximum detail
- Text and visual analysis are combined into enriched documents with structured format

### 2. Vectorization
- Documents are embedded using Nomic Embed Text
- Embeddings are stored in ChromaDB for fast similarity search

### 3. Retrieval
- MMR algorithm retrieves diverse, relevant chunks (k=8 for better context)
- Reduces bias by selecting from different document sections
- Ensures comprehensive context coverage
- Prioritizes OCR-extracted content for accurate answers

### 4. Generation
- Gemma 3 generates responses using retrieved context (4096 token window)
- Elite Chain-of-Thought prompting ensures direct, accurate, multilingual answers
- System prompt prioritizes OCR blocks for document understanding
- Streaming interface provides real-time feedback

---

## 🎨 Features in Detail

### Multimodal Processing
The system doesn't just read text—it understands visual content with aggressive OCR:
- **Text Extraction:** ALL visible text including headers, titles, university names, course codes
- **Diagrams:** Extracts structure, nodes, arrows, labels, and connections with flow explanation
- **Charts:** Reads data, trends, axis labels, and visual hierarchy
- **Formulas:** Transcribes mathematical expressions exactly
- **Tables:** Understands structured information and relationships
- **Two-Phase Output:** Structured OCR blocks + Visual descriptions for maximum accuracy

### Smart Context Management
- **4096 token context window** for handling long documents
- **MMR retrieval** prevents over-reliance on similar chunks
- **Metadata preservation** tracks source files and page numbers
- **Optimized chunking** (600 chars, 150 overlap) for focused retrieval
- **Enhanced retrieval** (k=8) for better context coverage

### Privacy & Security
- **100% local processing** - no data sent to external APIs
- **Offline operation** - works without internet after setup
- **No telemetry** - completely private
- **Smart caching** - processed documents cached locally for faster re-processing

### Performance Optimizations
- **Smart Caching:** MD5-based cache system skips Vision processing on cache hits
- **Windows Compatibility:** Special handling for file locking issues
- **Memory Management:** Garbage collection and resource cleanup
- **Modular Architecture:** Easy to extend and maintain

---

## 🐛 Troubleshooting

### Poppler Not Found (Windows)
If you get `poppler` errors:
1. Download Poppler from the official site
2. Extract to `C:\poppler` (or any location)
3. Add `C:\poppler\Library\bin` to your system PATH
4. Restart your terminal/IDE

### Ollama Connection Error
Ensure Ollama is running:
```bash
# Check if Ollama is running
ollama list

# If not, start Ollama service
ollama serve
```

### Model Not Found
If you get model errors, ensure models are pulled:
```bash
ollama pull gemma3:4b
ollama pull llama3.2-vision
ollama pull nomic-embed-text
```

### Memory Issues
If you encounter out-of-memory errors:
- Reduce `chunk_size` in `rag_engine.py`
- Use smaller models (e.g., `gemma3:2b` instead of `4b`)
- Process fewer PDFs at once

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **LangChain** for the excellent RAG framework
- **Ollama** for local LLM inference
- **Streamlit** for the beautiful web interface
- **ChromaDB** for efficient vector storage

---

## 📧 Contact

For questions, suggestions, or support, please open an issue on GitHub.

---

<div align="center">

**Built with ❤️ for students and researchers who value privacy**

⭐ Star this repo if you find it useful!

</div>

