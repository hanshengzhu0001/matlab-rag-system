#!/bin/bash
# MATLAB Vision RAG Setup Script
# Complete installation and setup guide for the vision-enhanced RAG system

set -e  # Exit on any error

echo "🚀 MATLAB Vision RAG Setup Script"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_step() {
    echo -e "${BLUE}[STEP $1]${NC} $2"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_warning "This script is optimized for macOS. Some commands may need adjustment."
fi

# Phase 0: Environment Setup
print_step "0" "Environment Setup"

# Check conda
if ! command -v conda &> /dev/null; then
    print_error "Conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Create/activate environment
ENV_NAME="matlab_rag"
if conda env list | grep -q "^${ENV_NAME} "; then
    print_success "Conda environment '${ENV_NAME}' already exists"
else
    print_step "0.1" "Creating conda environment"
    conda create -n ${ENV_NAME} python=3.11 -y
fi

print_step "0.2" "Activating environment and installing dependencies"
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Install Python dependencies
pip install -r requirements.txt

# Check Ollama
print_step "0.3" "Checking Ollama installation"
if ! command -v ollama &> /dev/null; then
    print_warning "Ollama not found. Installing..."
    brew install ollama
else
    print_success "Ollama is installed"
fi

# Check if Ollama models are available
echo "Checking Ollama models..."
if ollama list | grep -q "qwen2.5-coder"; then
    print_success "Qwen2.5-Coder model is available"
else
    print_warning "Qwen2.5-Coder model not found. Pulling..."
    ollama pull qwen2.5-coder:3b-instruct
fi

# Start Ollama server
print_step "0.4" "Starting Ollama server"
if pgrep -f "ollama serve" > /dev/null; then
    print_success "Ollama server is already running"
else
    ollama serve &
    sleep 3
    print_success "Ollama server started"
fi

# Phase 1: Build Visual Knowledge Base
print_step "1" "Building Visual Knowledge Base"

# Check if MATLAB is available
if ! command -v matlab &> /dev/null; then
    print_warning "MATLAB not found in PATH. OCR step will be skipped."
    print_warning "To enable OCR, ensure MATLAB is installed and in PATH."
    SKIP_OCR=true
else
    SKIP_OCR=false
fi

# Step 1.1: OCR Processing
if [ "$SKIP_OCR" = false ]; then
    print_step "1.1" "Running MATLAB OCR on documentation images"
    echo "This may take several minutes..."
    matlab -batch "run('extract_ocr.m')"
    print_success "OCR processing completed"
else
    print_warning "Skipping OCR - MATLAB not available"
    # Create empty OCR file for BLIP-only processing
    echo "[]" > image_ocr_data.json
fi

# Step 1.2: BLIP Captioning
print_step "1.2" "Generating BLIP descriptions for images"
python generate_blip_descriptions.py
print_success "BLIP captioning completed"

# Step 1.3: Build Enhanced Database
print_step "1.3" "Building enhanced RAG database with visual knowledge"
python -c "
from build_rag import MATLABRAGBuilder
builder = MATLABRAGBuilder(docs_path='matlab_documents/matlab', persist_dir='./chroma_db_matlab')
builder.build_database(max_files=None, include_visual=True)
print('✅ Enhanced database built with visual knowledge!')
"

# Phase 2: Test Vision Interface
print_step "2" "Testing Vision RAG Interface"

print_step "2.1" "Running basic functionality test"
python -c "
from vision_rag_core import VisionRAGAnalyzer
from query_rag import MATLABQuerySystem
print('Testing VisionRAGAnalyzer initialization...')
try:
    query_system = MATLABQuerySystem()
    analyzer = VisionRAGAnalyzer(query_system.chain)
    print('✅ Vision RAG system initialized successfully!')
except Exception as e:
    print(f'❌ Initialization failed: {e}')
    exit(1)
"

# Phase 3: Launch Interface
print_step "3" "Launching Vision RAG Web Interface"

print_success "Setup completed successfully!"
echo ""
echo "🎉 MATLAB Vision RAG Assistant is ready!"
echo ""
echo "📋 Next Steps:"
echo "   1. Launch the vision interface: python app_vision.py"
echo "   2. Or use the regular interface: python app.py"
echo "   3. Upload MATLAB images and ask questions!"
echo ""
echo "🌐 Interfaces:"
echo "   • Vision RAG: http://localhost:7860 (when launched)"
echo "   • Regular RAG: http://localhost:5000 (when launched)"
echo ""
echo "📚 What you can do:"
echo "   • Upload plots, diagrams, error messages"
echo "   • Ask 'What function creates this plot?'"
echo "   • Ask 'How do I modify this code?'"
echo "   • Get answers from 171K+ text chunks + 9K+ images"
echo ""

# Offer to launch immediately
read -p "🚀 Launch Vision RAG interface now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Launching Vision RAG interface..."
    python app_vision.py
fi
