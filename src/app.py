#!/usr/bin/env python3
"""
Flask backend API server for MATLAB RAG system
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import logging
from pathlib import Path

from query_rag import MATLABQuerySystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)  # Enable CORS for frontend

# Initialize the MATLAB RAG system (singleton)
query_system = None

def get_query_system():
    """Initialize and return the MATLAB query system (singleton pattern)."""
    global query_system
    if query_system is None:
        logger.info("Initializing MATLAB RAG system...")
        query_system = MATLABQuerySystem()
    return query_system


@app.route('/')
def index():
    """Serve the main frontend page."""
    return send_from_directory('static', 'index.html')


@app.route('/api/query', methods=['POST'])
def query():
    """Handle MATLAB query requests."""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                'success': False,
                'error': 'Question is required'
            }), 400
        
        logger.info(f"Processing query: {question[:100]}...")
        
        # Get the query system and process the question
        system = get_query_system()
        result = system.query(question, show_context=False)
        
        return jsonify({
            'success': result.get('success', True),
            'answer': result.get('answer', ''),
            'query_time': result.get('query_time', 0),
            'question': question
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        system = get_query_system()
        return jsonify({
            'status': 'healthy',
            'system_ready': True
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'system_ready': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    # Get port from environment or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    logger.info(f"Starting MATLAB RAG API server on port {port}")
    logger.info(f"Frontend will be available at: http://localhost:{port}")
    
    # Run in development mode (use gunicorn or uwsgi for production)
    app.run(host='0.0.0.0', port=port, debug=True)

