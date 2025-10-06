# 🤖 LangGraph Agentic RAG System

A comprehensive agentic RAG (Retrieval-Augmented Generation) system built with LangGraph, ChromaDB, and Google Gemini LLM, featuring advanced document processing, persistent memory, and intelligent web search capabilities.

## ✨ Features

- **🧠 Intelligent Agent Workflow**: Built with LangGraph for complex agentic behaviors and decision-making
- **📚 Advanced Document Processing**: ETL pipeline for PDF, DOCX, CSV, HTML, MD with intelligent chunking
- **🗄️ Vector Database**: ChromaDB with sentence transformers for semantic search and retrieval
- **🧠 Persistent Memory**: SQLite-based conversation history and user context management
- **🌐 Enhanced Web Search**: DuckDuckGo integration with result processing and summarization
- **🎯 Context-Aware Responses**: Multi-source information synthesis with confidence scoring
- **📊 Streaming Responses**: Real-time response generation with source attribution
- **🔄 RLHF Integration**: Comprehensive feedback collection and analysis system
- **📱 Modern UI**: Clean Streamlit interface with document upload and system monitoring
- **🔧 Function Calling**: Built-in calculator, datetime, and extensible tools
- **📈 Analytics Dashboard**: Performance metrics and improvement suggestions

## 🏗️ Architecture

```
├── src/
│   ├── agents/           # LangGraph workflow and memory management
│   ├── database/         # ChromaDB vector database management
│   ├── tools/           # Web search, RAG retrieval, function calling
│   └── utils/           # Gemini LLM client, document processing, RLHF
├── frontend/            # Streamlit web interface
├── config/              # Configuration management
└── data/               # Persistent storage (auto-created)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google API Key for Gemini LLM ([Get one here](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone or download the project**
2. **Run the setup script:**
   ```bash
   setup.bat
   ```

3. **Configure your API key:**
   - Edit `.env` file
   - Add your Google API key:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

4. **Start the application:**
   ```bash
   run.bat
   ```

5. **Open your browser** to `http://localhost:8501`

### Manual Setup (Alternative)

```bash
# Create virtual environment
python -m venv langraph_env
langraph_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
copy .env.example .env
# Edit .env with your API key

# Run the application
streamlit run frontend\app.py
```

## 📋 Usage Guide

### 1. Document Upload
- Click **"Browse files"** in the sidebar
- Upload PDF, DOCX, TXT, CSV, XLSX, HTML, or MD files
- Click **"Process Documents"** to add them to the knowledge base

### 2. Chat Interface
- Type your questions in the chat input
- The agent will automatically:
  - Search the web for current information
  - Query your uploaded documents
  - Perform calculations or function calls
  - Maintain conversation context

### 3. Settings & Controls
- **Streaming**: Toggle real-time response streaming
- **Tool Selection**: Enable/disable web search and knowledge base
- **Memory Management**: View and clear conversation history
- **Knowledge Base**: Monitor document count and clear database

### 4. Feedback System
- Rate responses when prompted (1-5 stars)
- Provide optional text feedback
- View feedback analytics in the dashboard

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Required
GOOGLE_API_KEY=your_google_api_key

# Optional Customization
MODEL_NAME=gemini-1.5-flash
TEMPERATURE=0.7
MAX_TOKENS=2048
CHUNK_SIZE=1000
MAX_MEMORY_MESSAGES=50
FEEDBACK_COLLECTION_RATE=0.3
```

### Supported File Types

- **PDF**: Text extraction with PyPDF2
- **DOCX**: Microsoft Word documents
- **TXT**: Plain text files
- **CSV/XLSX**: Data files (converted to text)
- **HTML**: Web pages (cleaned text)
- **MD**: Markdown files

## 🎯 Key Components

### LangGraph Workflow
- **Query Analysis**: Determines what tools are needed
- **Web Search**: Real-time information retrieval
- **RAG Retrieval**: Knowledge base querying
- **Function Calling**: Calculator, datetime, extensible
- **Response Synthesis**: Combines all information sources
- **Memory Update**: Maintains conversation context

### ChromaDB Integration
- Persistent vector storage
- Semantic search with embeddings
- Document chunking and metadata
- Similarity-based retrieval

### RLHF System
- User feedback collection
- Response quality tracking
- Performance analytics
- Continuous improvement insights

## 📊 Monitoring & Analytics

Access the feedback dashboard to view:
- Average response ratings
- Source usage statistics
- Response time metrics
- Improvement suggestions
- Rating distribution

## 🔌 Extending the System

### Adding New Tools
```python
# In src/tools/agentic_tools.py
def my_custom_function(param: str) -> str:
    # Your logic here
    return result

# Register the function
tools_manager.function_calling.register_function(
    name="my_function",
    func=my_custom_function,
    description="What this function does",
    parameters={"type": "object", "properties": {...}}
)
```

### Custom Document Processors
```python
# In src/utils/document_processor.py
def _process_custom_format(self, file_content: bytes) -> str:
    # Your processing logic
    return extracted_text

# Add to supported_extensions
self.supported_extensions['.custom'] = self._process_custom_format
```

## 🐛 Troubleshooting

### Common Issues

1. **"Google API key is required"**
   - Ensure `.env` file exists with valid `GOOGLE_API_KEY`

2. **ChromaDB errors**
   - Delete `data/chromadb` folder and restart

3. **Import errors**
   - Ensure virtual environment is activated
   - Run `pip install -r requirements.txt`

4. **Port already in use**
   - Change port in `.env`: `STREAMLIT_SERVER_PORT=8502`

### Performance Optimization

- Adjust `CHUNK_SIZE` for better document processing
- Lower `MAX_MEMORY_MESSAGES` for faster responses
- Increase `TEMPERATURE` for more creative responses

## 📝 Development

### Project Structure
```
langraph_agentic_rag/
├── src/                    # Core application code
│   ├── agents/            # LangGraph workflows
│   ├── database/          # ChromaDB management
│   ├── tools/            # Agentic tools
│   └── utils/            # Utilities and helpers
├── frontend/              # Streamlit interface
├── config/               # Configuration files
├── data/                 # Runtime data storage
├── requirements.txt      # Python dependencies
├── setup.bat            # Windows setup script
└── run.bat             # Windows run script
```

### Key Dependencies
- `langgraph==0.0.32` - Agentic workflow framework
- `chromadb==0.4.22` - Vector database
- `streamlit==1.30.0` - Web interface
- `google-generativeai==0.4.0` - Gemini LLM
- `sentence-transformers==2.2.2` - Text embeddings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangGraph** for the agentic workflow framework
- **ChromaDB** for vector database capabilities
- **Google Gemini** for the LLM backend
- **Streamlit** for the web interface
- **Sentence Transformers** for embeddings

## 📞 Support

For questions and support:
- Check the troubleshooting section
- Review configuration options
- Ensure all dependencies are installed correctly

---

**Happy Building! 🚀**

Built with ❤️ using LangGraph, ChromaDB, and Gemini LLM