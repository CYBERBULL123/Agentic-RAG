# 🚀 Quick Start Guide

## Setup Instructions

### 1. Prerequisites
- **Python 3.8+** installed on your system
- **Google API Key** for Gemini LLM ([Get one here](https://makersuite.google.com/app/apikey))
- **Internet connection** for web search and package installation

### 2. Installation

#### Option A: Automated Setup (Recommended)
```bash
# Run the setup script
setup.bat

# Edit the .env file with your API key
# Then run the application
run.bat
```

#### Option B: Manual Setup
```bash
# 1. Create virtual environment
python -m venv langraph_env
langraph_env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup environment
copy .env .env
# Edit .env and add your Google API key

# 4. Create directories
mkdir data\chromadb
mkdir data\logs

# 5. Run the application
streamlit run frontend\app.py
```

### 3. Configuration

Edit `.env` file with your settings:
```bash
# Required - Get from https://makersuite.google.com/app/apikey
GOOGLE_API_KEY=your_google_api_key_here

# Optional - Customize as needed
MODEL_NAME=models/gemini-2.5-flash
TEMPERATURE=0.7
CHUNK_SIZE=1000
MAX_MEMORY_MESSAGES=50
ENABLE_RLHF=true
```

---

## 🎯 Usage Guide

### Document Upload and Processing

1. **Supported File Types**
   - PDF documents
   - Microsoft Word (.docx)
   - Text files (.txt)
   - CSV data files
   - Excel spreadsheets (.xlsx)
   - HTML pages
   - Markdown files (.md)

2. **Upload Process**
   - Click "Browse files" in sidebar
   - Select multiple documents
   - Click "Process Documents"
   - Documents are cleaned, chunked, and embedded automatically
   - View processing status in real-time

### Chat Interface

1. **Basic Questions**
   ```
   "What is machine learning?"
   "Explain quantum computing"
   "How do neural networks work?"
   ```

2. **Document-Based Queries** (after uploading documents)
   ```
   "Summarize the uploaded research paper"
   "What are the key findings in my documents?"
   "Compare the methodologies in the papers"
   ```

3. **Web-Enhanced Queries**
   ```
   "What are the latest AI developments?"
   "Current trends in renewable energy"
   "Recent news about space exploration"
   ```

4. **Calculations and Functions**
   ```
   "Calculate 15% of 2500"
   "What time is it now?"
   "Compute (45 + 67) * 2.5"
   ```

5. **Mixed Queries** (Web + Documents + Memory)
   ```
   "How do current AI trends compare to what's in my research notes?"
   "Based on our previous conversation and recent news, what should I focus on?"
   ```

### Advanced Features

#### Memory Management
- **Persistent Conversations**: All chat history is automatically saved
- **Context Awareness**: Agent remembers previous discussions
- **User Profiles**: System learns your preferences over time
- **Clear Memory**: Use sidebar button to reset conversation

#### Feedback System
- **Rate Responses**: 1-5 star rating system appears periodically
- **Provide Comments**: Optional text feedback for improvements
- **View Analytics**: Check feedback dashboard for insights

#### System Monitoring
- **Knowledge Base Stats**: See document count and file types
- **Memory Usage**: Monitor conversation history
- **Processing Steps**: View how queries were handled
- **Source Attribution**: Know which sources were used

---

## 🔧 Advanced Configuration

### Document Processing Settings

```bash
# In .env file
CHUNK_SIZE=1000          # Text chunk size for embeddings
CHUNK_OVERLAP=200        # Overlap between chunks
```

### Memory Configuration

```bash
MAX_MEMORY_MESSAGES=50   # Max messages per session
CONTEXT_WINDOW=32000     # Context window size
```

### Model Settings

```bash
TEMPERATURE=0.7          # Response creativity (0-1)
MAX_TOKENS=2048         # Maximum response length
TOP_P=0.95              # Nucleus sampling parameter
```

### RLHF Settings

```bash
ENABLE_RLHF=true                    # Enable feedback collection
FEEDBACK_COLLECTION_RATE=0.3        # Rate of feedback requests (0-1)
```

---

## 🐛 Troubleshooting

### Common Issues

1. **"Google API key is required"**
   - Solution: Edit `.env` file and add valid `GOOGLE_API_KEY`
   - Get key from: https://makersuite.google.com/app/apikey

2. **ChromaDB errors**
   - Solution: Delete `data/chromadb` folder and restart application
   - The system will recreate the database automatically

3. **Import errors**
   - Solution: Ensure virtual environment is activated
   - Run: `pip install -r requirements.txt`

4. **Port already in use**
   - Solution: Change port in `.env`: `STREAMLIT_SERVER_PORT=8502`
   - Or kill existing processes using port 8501

5. **Document processing fails**
   - Check file format is supported
   - Ensure file is not corrupted
   - Try smaller files first

6. **Web search not working**
   - Check internet connection
   - DuckDuckGo might be temporarily unavailable
   - System will continue with knowledge base only

### Performance Optimization

- **Faster Processing**: Reduce `CHUNK_SIZE` to 500-800
- **Better Memory**: Increase `MAX_MEMORY_MESSAGES` to 100
- **More Creative**: Increase `TEMPERATURE` to 0.8-0.9
- **More Focused**: Decrease `TEMPERATURE` to 0.3-0.5

### Data Management

- **Clear Old Data**: Use sidebar buttons to clear memory/knowledge base
- **Backup**: Copy `data/` folder to backup conversations and documents
- **Reset**: Delete `data/` folder to completely reset system

---

## 📊 System Architecture

```
📦 LangGraph Agentic RAG System
├── 🧠 Agent Workflow (LangGraph)
│   ├── Query Analysis
│   ├── Web Search
│   ├── RAG Retrieval  
│   ├── Function Calling
│   └── Response Synthesis
├── 📚 Knowledge Base (ChromaDB)
│   ├── Document Processing
│   ├── Text Chunking
│   ├── Embeddings (Sentence Transformers)
│   └── Similarity Search
├── 🧠 Memory System (SQLite)
│   ├── Conversation History
│   ├── User Profiles
│   └── Context Knowledge
├── 🔄 RLHF System
│   ├── Feedback Collection
│   ├── Analytics
│   └── Improvement Tracking
└── 🖥️ Web Interface (Streamlit)
    ├── Chat Interface
    ├── Document Upload
    ├── System Monitoring
    └── Analytics Dashboard
```

---

## 🎯 Example Use Cases

### 1. Research Assistant
Upload research papers, ask questions about findings, get summaries, and compare methodologies across documents.

### 2. Study Companion  
Upload textbooks and lecture notes, ask for explanations, get practice questions, and track learning progress.

### 3. Business Intelligence
Upload reports and data files, ask for insights, get trend analysis, and generate summaries for stakeholders.

### 4. Technical Documentation
Upload manuals and guides, ask how-to questions, get troubleshooting help, and find specific procedures.

### 5. Creative Writing
Upload story drafts, get feedback on plot and characters, brainstorm ideas, and improve writing style.

---

## 🚀 Next Steps

1. **Upload Documents**: Start with a few test documents
2. **Explore Features**: Try different types of queries
3. **Provide Feedback**: Help improve the system with ratings
4. **Customize Settings**: Adjust configuration for your needs
5. **Monitor Performance**: Use the analytics dashboard

---

## 🆘 Support

- **Issues**: Check troubleshooting section above
- **Configuration**: Review environment variables
- **Performance**: Try optimization suggestions
- **Features**: Explore the examples and use cases

**Happy exploring! 🎉**