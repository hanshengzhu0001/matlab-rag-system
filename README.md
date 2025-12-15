# MATLAB-RAG-System

An intelligent RAG (Retrieval-Augmented Generation) system for MATLAB documentation that provides contextual code assistance using CodeLlama.

## Features

- **Document Processing**: Automated parsing and chunking of MATLAB documentation
- **Vector Embeddings**: BGE-base-en-v1.5 embeddings for semantic search
- **Hybrid Retrieval**: Combines dense (semantic) and sparse (keyword) search
- **Code Generation**: CodeLlama integration for MATLAB code assistance
- **CPU/GPU Support**: Optimized for both laptop and workstation deployment

## Architecture

- `build_rag.py`: Database builder with optimized batch processing
- `query_rag.py`: Query interface with CodeLlama integration
- Custom embedding classes for compatibility across different hardware
- ChromaDB for efficient vector storage and retrieval

## Quick Start

```bash
# Set up environment
conda create -n matlab_rag python=3.11 -y
conda activate matlab_rag
pip install langchain langchain-community langchain-chroma pypdf beautifulsoup4 sentence-transformers chromadb openai

# Optional: Set up OpenAI API for code generation
export OPENAI_API_KEY="your-openai-api-key-here"

# Build the database
python build_rag.py

# Query the system (interactive)
python query_rag.py

# Or use the test script
python test_query.py
```

## Performance

- Processes 12,000+ MATLAB documentation files
- Creates 170,000+ searchable chunks
- **DeepSeek API**: ~30-second query response times
- **Ollama Local**: ~4-minute query response times (resource intensive)
- Optimized for both CPU and GPU workflows

## Example Results

### Query: "How do I use the plot function to create a line graph with custom colors?"

**Response Time**: ~29 seconds  
**Method**: RAG + DeepSeek API (MATLAB documentation retrieval + code generation)

```matlab
% Create a line graph with custom colors using MATLAB's plot function
% This script demonstrates multiple ways to specify custom colors

%% Method 1: Using RGB triplets (most common approach)
x = 1:10;
y1 = sin(x);
y2 = cos(x);

figure('Position', [100, 100, 800, 600]);
subplot(2, 2, 1);
plot(x, y1, 'LineWidth', 2, 'Color', [0.5, 0, 0.8]); % Purple color
hold on;
plot(x, y2, 'LineWidth', 2, 'Color', [0, 0.7, 0.3]); % Greenish color
legend('sin(x)', 'cos(x)');
hold off;

%% Method 2: Using hexadecimal color codes
subplot(2, 2, 2);
plot(x, y1, 'LineWidth', 2, 'Color', '#00841a'); % Dark green

%% Method 3: Using color names
subplot(2, 2, 3);
plot(x, y1, 'LineWidth', 2, 'Color', 'red');

%% Method 4: Combining line specification and color properties
subplot(2, 2, 4);
p = plot(x, y1, '-o', 'Color', [0.2, 0.4, 0.8], 'LineWidth', 2, 'MarkerSize', 8);
set(p, 'LineStyle', '--', 'Marker', '*', 'Color', '#FF5733');
```

*Generated complete MATLAB script with 4 different color methods, error handling, and comprehensive documentation.*

## Use Cases

- MATLAB code assistance and debugging
- Function discovery and usage examples
- Technical documentation search
- Code generation from natural language descriptions

## Technology Stack

- **Embeddings**: BAAI/bge-base-en-v1.5 (768-dim)
- **LLM**: GPT-4o-mini via OpenAI API (or CodeLlama via Ollama)
- **Vector DB**: ChromaDB with HNSW indexing
- **Framework**: LangChain for RAG orchestration
- **Processing**: Sentence Transformers + PyTorch

## Code Generation Setup

### Option 1: DeepSeek API (Recommended for Hong Kong)
```bash
# Get API key from: https://platform.deepseek.com/api_keys
export DEEPSEEK_API_KEY="your-deepseek-api-key-here"

# Test the API
python test_deepseek_api.py

# Run the RAG demo
python demo_openai.py
```

**Why DeepSeek?** Fast, reliable, and works in Hong Kong ✅

### Option 2: OpenAI API
```bash
# Get API key from: https://platform.openai.com/api-keys
export OPENAI_API_KEY="your-openai-api-key-here"

# Test the system
python demo_openai.py
```

**Note for Hong Kong users:** OpenAI API has regional restrictions ❌

### Option 3: Local Ollama (Resource Intensive)
```bash
# Install and start Ollama
brew install ollama
ollama serve

# Pull CodeLlama model (in another terminal)
ollama pull codellama:7b

# Test the system (may overheat laptop)
python demo_openai.py
```

**Performance Comparison:**
- **DeepSeek API**: ⚡ Fast, ✅ Works in Hong Kong, 💰 Cost-effective
- **OpenAI API**: ⚡ Fast, ❌ Restricted in Hong Kong, 💰 Standard pricing
- **Ollama Local**: 🐌 Very slow on laptop CPU, 🔥 May cause overheating

## Project Structure

```
matlab_rag/
├── build_rag.py          # Database builder
├── query_rag.py          # Query interface
├── test_rag_components.py # Component testing
├── scan_file_types.py     # Document analysis
├── simple_html_test.py    # HTML parsing tests
├── README.md             # This file
├── .gitignore           # Git ignore rules
└── matlab_documents/    # MATLAB documentation (not in repo)
```

## Development Notes

- Custom BGE embedding class bypasses LangChain wrapper compatibility issues
- Batch processing (100 chunks/batch) prevents memory overflow
- CPU-optimized settings for laptop deployment
- Progress logging for long-running operations
