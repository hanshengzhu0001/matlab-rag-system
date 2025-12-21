# MATLAB-RAG-System

An intelligent multimodal RAG (Retrieval-Augmented Generation) system for MATLAB documentation that combines text and vision analysis for comprehensive code assistance.

## ✨ Features

### Core Capabilities
- **📄 Text Analysis**: Full MATLAB documentation RAG with CodeLlama/Qwen integration
- **🖼️ Vision Analysis**: BLIP-powered image understanding for plots, diagrams, and UI
- **🔍 Hybrid Retrieval**: Dense + sparse search across 171K+ text chunks + 9K+ images
- **⚡ Real-time Processing**: GPU-accelerated inference with RTX 5090 optimization
- **🌐 Web Interfaces**: Modern UIs for both text and vision queries

### Multimodal Knowledge Base
- **Text Pipeline**: HTML documentation → BGE embeddings → ChromaDB
- **Vision Pipeline**: Images → MATLAB OCR + BLIP captions → BGE embeddings → ChromaDB
- **Unified Retrieval**: Single query searches across all knowledge types

## Architecture

- `build_rag.py`: Database builder with optimized batch processing
- `query_rag.py`: Query interface with CodeLlama integration
- Custom embedding classes for compatibility across different hardware
- ChromaDB for efficient vector storage and retrieval

## 🚀 Quick Start

### Option 1: Text-Only RAG (Original)
```bash
# Basic setup
conda create -n matlab_rag python=3.11 -y
conda activate matlab_rag
pip install -r requirements.txt

# Build text database
python build_rag.py

# Launch text interface
python app.py  # http://localhost:5000
```

### Option 2: Full Vision RAG (Recommended)
```bash
# Complete automated setup
./setup_vision_rag.sh

# Or manual setup:
# 1. Install vision dependencies
pip install Pillow opencv-python-headless transformers torch accelerate gradio

# 2. Process visual knowledge
matlab -batch "run('extract_ocr.m')"          # OCR (optional)
python generate_blip_descriptions.py          # BLIP captions
python build_rag.py --include-visual          # Enhanced database

# 3. Launch vision interface
python app_vision.py  # http://localhost:7860
```

### Command Line Usage
```bash
# Text queries
python query_rag.py

# Demo with examples
python demo_openai.py
```

### Web Interface Setup

```bash
# Make sure Ollama is running (if using local Qwen model)
ollama serve

# In another terminal, start the web server
python app.py
```

## Performance

- Processes 12,000+ MATLAB documentation files
- Creates 170,000+ searchable chunks
- **Qwen2.5-Coder-3B**: ~15-30 second query response times (GPU accelerated)
- **BLIP-2 Vision**: ~5-10 second image analysis (GPU accelerated)
- **OpenAI API**: ~20-40 second query response times
- Optimized for RTX 5090 GPU acceleration

## Example Results

### Query: "How do I use the plot function to create a line graph with custom colors?"

**Response Time**: ~20 seconds  
**Method**: RAG + Qwen2.5-Coder-3B (MATLAB documentation retrieval + code generation)

**Search Process**:
- **Semantic Search**: BGE-base-en-v1.5 embeddings search through 171,366 documentation chunks
- **Vector Database**: ChromaDB stores embeddings of MATLAB HTML documentation chunks (600 chars with 100 overlap)
- **Retrieval**: Top 5 most relevant chunks retrieved based on semantic similarity to query

**Database Contents**: Vector embeddings of parsed MATLAB documentation including:
- Function reference pages
- Plotting and graphics documentation
- Color specification syntax
- Code examples and best practices
- Error handling patterns

**Code Generation**: DeepSeek API receives retrieved MATLAB documentation as context, combines it with its knowledge to generate accurate, documentation-compliant code.

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

## 🎯 Use Cases

### Text-Based Queries
- MATLAB code assistance and debugging
- Function discovery and usage examples
- Technical documentation search
- Code generation from natural language descriptions

### Vision-Based Queries (New!)
- **Plot Analysis**: Upload a MATLAB plot → "What function creates this 3D surface?"
- **Diagram Understanding**: Upload flowcharts → "How do I implement this workflow?"
- **UI Guidance**: Upload MATLAB interface screenshots → "How do I access this toolbox?"
- **Error Resolution**: Upload error messages → "What does this error mean?"
- **Code Recreation**: Upload visualizations → "How do I recreate this plot?"

### BLIP-2 Advanced Visual Question Answering
**Now Works With:**
- ✅ **Trigonometric plots**: sin, cos, tan functions and combinations
- ✅ **Polynomial functions**: linear, quadratic, cubic, power functions
- ✅ **Exponential/Logarithmic**: growth/decay curves, semi-log plots
- ✅ **Statistical visualizations**: histograms, box plots, scatter plots
- ✅ **Chart types**: bar charts, pie charts, area plots
- ✅ **Multi-dimensional**: 3D surfaces, contour plots, heatmaps
- ✅ **Time series**: temporal data, signal processing plots
- ✅ **Complex diagrams**: multi-panel figures, technical schematics

**Enhanced Capabilities:**
- 🎯 **VQA-Powered Recognition**: Uses BLIP-2's Visual Question Answering for specific function identification
- 🎯 **Mathematical Question Answering**: Asks targeted questions about plotted functions
- 🎯 **Intelligent Answer Selection**: Chooses most specific and mathematically accurate responses
- 🎯 **Trigonometric Expertise**: Superior recognition of sine, cosine, and trigonometric combinations
- 📝 **Context-Aware Descriptions**: Provides plot-type-specific MATLAB context
- 🔍 **Multi-Question Strategy**: Asks multiple VQA questions for comprehensive analysis
- 🧹 **Answer Quality Filtering**: Eliminates generic responses, prioritizes specific functions
- ⚡ **Optimized Performance**: FP16 acceleration on RTX GPUs

**Recognition Examples:**
- **Trigonometric**: `"sine and cosine combination"`, `"trigonometric functions"`
- **Polynomial**: `"quadratic function"`, `"linear relationship"`
- **Exponential**: `"exponential growth function"`, `"logarithmic scale"`
- **Plot Types**: `"scatter plot"`, `"bar chart"`, `"histogram"`, `"surface plot"`
- **Statistical**: `"box plot"`, `"violin plot"`, `"distribution histogram"`
- **Multi-dimensional**: `"3D surface plot"`, `"contour plot"`, `"heatmap"`

**VQA Analysis Approach:**
- **Question 1**: "What mathematical function is shown in this plot?"
- **Question 2**: "Is this a trigonometric function? If so, which one?"
- **Question 3**: "What type of mathematical relationship is plotted?"
- **Answer Selection**: Prioritizes specific function names over generic descriptions
- **Fallback Strategy**: Uses basic captioning if VQA fails

**Quality Improvements:**
- **Superior Specificity**: BLIP-2's VQA provides more precise mathematical identification
- **Trigonometric Excellence**: Better recognition of sine/cosine patterns than BLIP-base
- **Question-Driven Analysis**: Targeted questioning yields better results than generic captioning
- **Technical Accuracy**: Precise mathematical terminology and function identification

**Example Improvements:**

**Before (Repetitive):**
```
"linear combination of two functions | linear combination of two functions | linear combination of two functions | linear combination of two functions..."
```

**After (Optimized):**
```
"A MATLAB plot showing a combination of sine and cosine functions. Mathematical functions: combination of sine and cosine functions. This appears to be a MATLAB-generated visualization."
```

**Complete Code Generation Example:**
```matlab
% Create time series data
t = 0:0.1:10;
y = sin(t) + 0.5*cos(2*t);

% Create the plot
figure;
plot(t, y, 'LineWidth', 2);
xlabel('Time');
ylabel('Amplitude');
title('Time Series Plot');
grid on;
```
*Generated from image analysis + MATLAB documentation retrieval*

**Key Fixes:**
- ✅ **No repetition**: Single descriptive statement instead of repeated phrases
- ✅ **Function identification**: Explicitly recognizes sin and cos functions
- ✅ **Concise output**: Informative and to-the-point descriptions
- ✅ **Technical context**: Maintains MATLAB-specific insights

**Remaining Limitations:**
- ⚠️ **Extremely abstract art** (non-technical images)
- ⚠️ **Corrupted or very low-quality images**
- ⚠️ **Real-time video** (static images only)

### Example Vision Queries
```
🖼️ Upload: [3D surface plot image]
❓ Question: "What MATLAB function generates this type of 3D surface plot?"
🤖 Answer: "This appears to be created with the `surf()` or `mesh()` functions..."

🖼️ Upload: [Error message screenshot]
❓ Question: "How do I fix this MATLAB error?"
🤖 Answer: "This Index exceeds matrix dimensions error typically occurs when..."

🖼️ Upload: [Color plot screenshot]
❓ Question: "How do I change the colormap of this plot?"
🤖 Answer: "You can modify the colormap using `colormap()` function..."
```

## 🛠️ Technology Stack

### Core Components
- **Embeddings**: BAAI/bge-base-en-v1.5 (768-dim) for both text and visual content
- **LLM**: Qwen2.5-Coder-3B via Ollama (code-specialized, GPU accelerated)
- **Fallback**: GPT-4o-mini via OpenAI API
- **Vector DB**: ChromaDB with HNSW indexing (unified text + vision)
- **Framework**: LangChain for RAG orchestration

### Vision Components
- **OCR Engine**: MATLAB's built-in OCR (optimized for technical screenshots)
- **Image Understanding**: Salesforce BLIP-2 Opt-2.7B (Hugging Face, VQA-powered)
  - **Capabilities**: Advanced image captioning + Visual Question Answering (VQA)
  - **Multi-strategy processing**: Technical prompts + VQA + intelligent combination
  - **Mathematical recognition**: Identifies sin, cos, tan, exp, log functions
  - **Quality optimization**: Smart deduplication + repetition elimination
  - **Best for**: Complex plots, technical diagrams, detailed UI analysis
  - **Performance**: 10x more detailed than base BLIP, understands MATLAB contexts
- **Image Processing**: OpenCV + Pillow for preprocessing
- **GPU Acceleration**: PyTorch with CUDA/ROCm support (FP16 optimized)

### Processing Pipeline
- **Text**: BeautifulSoup HTML parsing → Recursive text splitting → BGE embeddings
- **Vision**: MATLAB OCR + BLIP captions → Combined text → BGE embeddings
- **Retrieval**: Hybrid dense + sparse search across unified vector space
- **Generation**: Context-enriched prompts → CodeLlama/Qwen generation

## Code Generation Setup

### Option 1: Qwen2.5-Coder-3B (Recommended - Local & Fast)
```bash
# Install Ollama (one-time setup)
brew install ollama

# Pull the optimized code model
ollama pull qwen2.5-coder:3b-instruct

# Start Ollama server
ollama serve

# Test the system
python demo_openai.py
```

**Why Qwen2.5-Coder?** ⚡ Fast local inference, 🎯 Code-specialized, 🚀 RTX 5090 accelerated, 🔒 No API keys needed

### Option 2: OpenAI API (Cloud Alternative)
```bash
# Get API key from: https://platform.openai.com/api-keys
export OPENAI_API_KEY="your-openai-api-key-here"

# Test the system
python demo_openai.py
```

**Performance Comparison:**
- **Qwen2.5-Coder-3B**: ⚡ ~15-30s, ✅ Local, 🎯 Code-optimized, 🚀 RTX 5090 accelerated
- **OpenAI API**: ⚡ ~20-40s, ☁️ Cloud, 💰 Pay-per-use, 🌐 Internet required

## 🌐 Web Interfaces

### Text RAG Interface (Original)
```bash
python app.py  # http://localhost:5000
```
- **Features**: Modern responsive UI, example queries, real-time code generation
- **API**: REST endpoints for text queries and health checks

### Vision RAG Interface (New)
```bash
python app_vision.py  # http://localhost:7860
```
- **Features**: Image upload, BLIP-powered analysis, multimodal answers
- **Capabilities**: Analyze MATLAB plots, diagrams, UI screenshots, error messages
- **Examples**: "What function creates this plot?", "How do I modify these colors?"

### API Endpoints (Text Interface)

- `GET /` - Web frontend
- `POST /api/query` - Submit MATLAB questions
  ```json
  {
    "question": "How do I create a matrix and plot it?"
  }
  ```
- `GET /api/health` - Health check endpoint

### Vision API (app_vision.py)
- `POST /api/analyze` - Upload image + question for multimodal analysis
- Returns: Image description + documentation-based answer + sources

## 📁 Project Structure

```
matlab_rag/
├── 🔧 Core System
│   ├── build_rag.py          # Database builder (text + vision)
│   ├── query_rag.py          # Text RAG system & CodeLlama integration
│   └── vision_rag_core.py    # Vision analysis engine
│
├── 🌐 Web Interfaces
│   ├── app.py                # Text RAG Flask server
│   ├── app_vision.py         # Vision RAG Gradio interface
│   └── static/
│       └── index.html       # Text RAG frontend
│
├── 🖼️ Vision Processing
│   ├── extract_ocr.m         # MATLAB OCR for documentation images
│   └── generate_blip_descriptions.py  # BLIP image captioning
│
├── 🧪 Testing & Demo
│   ├── demo_openai.py        # Demo script
│   ├── test_rag_components.py # Component testing
│   └── scan_file_types.py     # Document analysis
│
├── 📦 Configuration
│   ├── requirements.txt       # Python dependencies
│   ├── setup_vision_rag.sh    # Complete setup script
│   └── .gitignore            # Git ignore rules
│
├── 📚 Documentation
│   └── README.md             # This file
│
└── 💾 Data (Generated)
    ├── chroma_db_matlab/     # Vector database (text + vision)
    ├── image_ocr_data.json   # OCR results
    ├── visual_knowledge.json # BLIP-enhanced knowledge
    └── matlab_documents/     # MATLAB docs (not in repo)
```

## Development Notes

- Custom BGE embedding class bypasses LangChain wrapper compatibility issues
- Batch processing (100 chunks/batch) prevents memory overflow
- CPU-optimized settings for laptop deployment
- Progress logging for long-running operations
