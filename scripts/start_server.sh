#!/bin/bash
# Startup script for MATLAB RAG web server

echo "🚀 Starting MATLAB RAG Web Server"
echo "=================================="
echo ""

# Check if Ollama is running (optional, but recommended)
if command -v ollama &> /dev/null; then
    echo "✅ Ollama found"
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama server is running"
    else
        echo "⚠️  Ollama server not running. Start it with: ollama serve"
        echo "   (Code generation will fallback to OpenAI if configured)"
    fi
else
    echo "⚠️  Ollama not found. Install with: brew install ollama"
fi

echo ""
echo "📡 Starting Flask server..."
echo "🌐 Frontend will be available at: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app.py

