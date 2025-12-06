# 🎓 UniStudyRAG: Local, Multimodal AI Study Assistant

<div align="center">

**Privacy-first RAG system running on local hardware (RTX 2070 laptop optimized). Combines Gemma 3 (4b) for reasoning and Llama 3.2 Vision for image analysis.**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://www.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 🚀 Key Features

### 🧠 Hybrid Architecture
Vision + Text pipeline that processes both textual content and visual elements (diagrams, charts, schematics) from PDF documents.

### 🔒 100% Local & Private
Runs completely offline using Ollama. Your documents never leave your machine. Perfect for sensitive academic materials and confidential research papers.

### 👁️ Multimodal RAG
Understands and analyzes:
- Text content from PDFs
- Diagrams and flowcharts
- Mathematical equations and formulas
- Tables and structured data
- Technical schematics

### ⚡ Smart Retrieval
Uses **MMR (Maximal Marginal Relevance)** algorithm to reduce retrieval bias. Ensures diverse context retrieval from different parts of documents, preventing the system from focusing only on similar chunks.

### 🗣️ Multilingual Chain-of-Thought
Advanced prompting strategy that:
- Handles German/English documents with Turkish (or any language) answers
- Distinguishes between document owners and references (critical for CVs and academic papers)
- Maintains technical terminology in original language while translating explanations

### 💬 Streaming Responses
ChatGPT-style streaming interface with real-time word-by-word response generation for a smooth user experience.

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

### 4. Install Ollama Models

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
├── rag_engine.py         # Core RAG logic (multimodal processing)
├── main.py               # CLI version (alternative interface)
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
├── README.md            # This file
├── data/                # PDF storage (gitignored)
└── chroma_db/          # Vector database (gitignored)
```

---

## 🔍 How It Works

### 1. Document Processing
- PDFs are loaded and split into manageable chunks
- Each page is converted to an image
- Vision model analyzes images for diagrams, charts, and visual content
- Text and visual analysis are combined into enriched documents

### 2. Vectorization
- Documents are embedded using Nomic Embed Text
- Embeddings are stored in ChromaDB for fast similarity search

### 3. Retrieval
- MMR algorithm retrieves diverse, relevant chunks
- Reduces bias by selecting from different document sections
- Ensures comprehensive context coverage

### 4. Generation
- Gemma 3 generates responses using retrieved context
- Chain-of-Thought prompting ensures accurate, multilingual answers
- Streaming interface provides real-time feedback

---

## 🎨 Features in Detail

### Multimodal Processing
The system doesn't just read text—it understands visual content:
- **Diagrams:** Extracts structure and relationships
- **Charts:** Reads data and trends
- **Formulas:** Captures mathematical expressions
- **Tables:** Understands structured information

### Smart Context Management
- **4096 token context window** for handling long documents
- **MMR retrieval** prevents over-reliance on similar chunks
- **Metadata preservation** tracks source files and page numbers

### Privacy & Security
- **100% local processing** - no data sent to external APIs
- **Offline operation** - works without internet after setup
- **No telemetry** - completely private

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

