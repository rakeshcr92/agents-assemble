from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from orchestrator_single import process_voice_sync
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({
        "message": "Life Witness Agent API",
        "version": "1.0.0",
        "endpoints": {
            "/api/voice/process": "POST - Process voice input",
            "/api/text/process": "POST - Process text input (for testing)",
            "/api/health": "GET - Health check"
        }
    })

@app.route('/api/voice/process', methods=['POST'])
def process_voice():
    """Process voice input"""
    try:
        data = request.json
        audio_base64 = data.get('audio')
        
        if not audio_base64:
            return jsonify({"error": "No audio data provided"}), 400
        
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_base64)
        
        # Process through orchestrator
        result = process_voice_sync(audio_data=audio_bytes)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/text/process', methods=['POST'])
def process_text():
    """Process text input (for testing without audio)"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Process through orchestrator
        result = process_voice_sync(test_text=text)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        "status": "healthy",
        "service": "life-witness-agent"
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)