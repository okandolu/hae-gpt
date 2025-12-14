# HAE RAG System: Cross-lingual Medical Question Answering

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready **Retrieval-Augmented Generation (RAG) system** for answering medical questions about Hereditary Angioedema (HAE). The system features cross-lingual capabilities, multi-mode generation, and state-of-the-art reasoning with DeepSeek R1.

## 🎯 Overview

This system addresses the challenge of providing accessible, accurate medical information about Hereditary Angioedema by:

1. **Cross-lingual Support**: Accepts queries in Turkish, retrieves from English medical literature, and responds in the query language
2. **Multi-Mode Generation**: Three tailored response modes for different audiences (patients, personalized care, academics)
3. **Reasoning Transparency**: Integrates DeepSeek R1 for explainable answer generation
4. **High-Quality Retrieval**: Uses BGE-M3 multilingual embeddings with FAISS for efficient semantic search

### Research Context

This system was developed as part of research in medical AI and cross-lingual information retrieval. It demonstrates:
- Effective cross-lingual RAG for low-resource languages in medical domains
- Multi-mode prompt engineering for audience-specific content generation
- Integration of reasoning-capable LLMs in production RAG systems
- Quality-assured medical information delivery with citation transparency

##  Key Features

### Core Capabilities

- **Cross-lingual RAG Pipeline**
  - Turkish query → English context retrieval → Turkish response
  - Language detection and automatic response adaptation
  - Maintains medical accuracy across language boundaries

- **Multi-Mode Generation**
  - **Patient Mode**: 8th-grade reading level, empathetic, jargon-free
  - **Personalized Mode**: Clinical data integration for individualized advice
  - **Academic Mode**: Technical depth with proper citations

- **DeepSeek R1 Reasoning**
  - Transparent reasoning process
  - Quality validation and self-correction
  - Mode-specific temperature tuning

- **🔍 Advanced Retrieval**
  - BGE-M3 (1024-dim) multilingual embeddings
  - FAISS IndexFlatIP for cosine similarity
  - GPT-4o-mini guided semantic chunking
  - Metadata-rich citation tracking

### Production Features

- **Batch Processing**: Process multiple queries with progress tracking and database persistence
- **Analytics Dashboard**: Token usage, similarity scores, query statistics
- **Streamlit UI**: Interactive web interface with real-time generation
- **Export Options**: JSON, Markdown, Excel formats
- **Optimized Performance**: Caching, batch embedding, efficient vector search

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Query (Turkish)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Retriever (BGE-M3 + FAISS)                      │
│  • Cross-lingual embedding                                   │
│  • Semantic search (cosine similarity)                       │
│  • Top-k context retrieval                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            Generator (DeepSeek R1)                           │
│  • Mode-specific prompts                                     │
│  • Reasoning generation                                      │
│  • Quality validation                                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         Response (Turkish) + Citations + Reasoning           │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Embeddings** | BGE-M3 (BAAI) | Multilingual semantic encoding |
| **Vector Store** | FAISS (IndexFlatIP) | Efficient similarity search |
| **LLM** | DeepSeek R1 | Reasoning-capable generation |
| **Framework** | LangChain | RAG pipeline orchestration |
| **Chunking** | GPT-4o-mini | Semantic document segmentation |
| **UI** | Streamlit | Interactive web interface |
| **DB** | SQLite | Query and result persistence |

## Installation

### Prerequisites

- Python 3.10 or higher
- API Keys:
  - OpenAI API key (for GPT-4o-mini chunking)
  - DeepSeek API key (for R1 generation)
  - HuggingFace token (optional, for model downloads)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/hae-rag-system.git
cd hae-rag-system
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here # used for chunking
DEEPSEEK_API_KEY=your_deepseek_api_key_here # used as an llm
HUGGINGFACE_API_KEY=your_hf_token_here  # optional, if you do not want use to use embedding model locally
```

### Step 5: Prepare Data

Place your PDF documents in `data/raw/`:

```bash
mkdir -p data/raw
# Add your PDF files here
```

### Step 6: Build Vector Store

```bash
python setup_pipeline.py
```

This will:
1. Extract text and tables from PDFs
2. Perform GPT-guided semantic chunking
3. Generate BGE-M3 embeddings
4. Build FAISS index
5. Save to `data/vectorstore/`

## 🚀 Quick Start

### Run Streamlit Interface

```bash
streamlit run streamlit_app.py
```

Access at `http://localhost:8501`

### Command Line Usage

```python
python query_system.py "what is HAE"
```

## 📖 Usage

### Single Query Mode

1. Select generation mode (Patient / Personalized / Academic)
2. Enter your question
3. Adjust retrieval parameters (top-k, threshold)
4. View answer with citations and reasoning
5. Export results (JSON, Markdown, or database)

### Batch Processing Mode

1. Navigate to "Toplu Soru-Cevap" tab
2. Choose input method:
   - **Manual Entry**: Type questions (one per line)
   - **File Upload**: Excel/TXT file with questions
3. Configure mode and parameters
4. Process batch
5. View results and export to Excel

### Personalized Mode

For personalized mode, provide clinical information:

```
HAE Tip 1, 35 yaş, son atak 2 hafta önce, İcatibant kullanıyor, gebelik planı var
```

The system will tailor responses to the specific patient profile.

## 🔧 System Components

### 1. Document Processor (`src/document_processor.py`)

- **PDF Extraction**: Text and table extraction with pdfplumber
- **GPT-Guided Chunking**: Semantic segmentation with GPT-4o-mini
- **Metadata Enrichment**: Section detection, table flagging, reference extraction

### 2. Embedding Service (`src/embedding_service.py`)

- **Model**: BGE-M3 (1024-dimensional)
- **Batch Processing**: Configurable batch size for efficiency
- **Normalization**: L2 normalization for cosine similarity

### 3. Vector Store (`src/vector_store.py`)

- **Index Type**: FAISS IndexFlatIP (inner product)
- **Persistence**: Index + metadata pickling
- **Search**: Top-k retrieval with similarity threshold

### 4. Retriever (`src/retriever.py`)

- **Query Embedding**: Real-time embedding generation
- **Context Formatting**: LLM-ready context string assembly
- **Citation Tracking**: Metadata preservation for attribution

### 5. Generator (`src/generator.py`)

- **Model**: DeepSeek R1 (deepseek-reasoner)
- **Mode-Specific Prompts**: 3 carefully engineered system prompts
- **Quality Validation**: Post-generation quality checks
- **Token Management**: Usage tracking and reporting

### 6. Citation Formatter (`src/citation_formatter.py`)

- **APA Style**: Academic citation formatting
- **Patient Style**: Simplified citation display
- **Metadata Integration**: Reference, year, publication details

## ⚙️ Configuration

### Key Configuration (`config.py`)
Checkout config.py for explanations

### Mode-Specific Prompts

Edit `config.py` to customize system prompts for each mode:
- `SYSTEM_PROMPT_PATIENT`
- `SYSTEM_PROMPT_PATIENT_PERSONALIZED`
- `SYSTEM_PROMPT_ACADEMIC`


## 📚 Project Structure

```
hae-rag-system/
├── src/                          # Core modules
│   ├── __init__.py
│   ├── document_processor.py     # PDF processing & chunking
│   ├── embedding_service.py      # BGE-M3 embeddings
│   ├── vector_store.py           # FAISS vector store
│   ├── retriever.py              # Retrieval logic
│   ├── generator.py              # DeepSeek R1 generation
│   ├── citation_formatter.py     # Citation formatting
│   └── gpt_chunker.py            # GPT-guided chunking
├── streamlit_app.py              # Streamlit UI
├── batch_processor.py            # Batch processing
├── batch_query_db.py             # SQLite database
├── config.py                     # Configuration
├── requirements.txt              # Dependencies
├── pyproject.toml                # Modern Python packaging
├── LICENSE                       # MIT License
├── CITATION.cff                  # Citation metadata
├── README.md                     # This file
└── data/
    ├── raw/                      # Input PDFs
    ├── processed/                # Processed chunks
    └── vectorstore/              # FAISS index + metadata
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This system is for research and educational purposes. Always consult healthcare professionals for medical advice.
