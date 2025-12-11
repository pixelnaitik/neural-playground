"""
Voice Studio routes - Text-to-Speech using gTTS library.
"""

import os
import uuid
import time
import threading
from flask import Blueprint, render_template, request, jsonify, current_app, send_file

audio_bp = Blueprint('audio', __name__)

# Supported languages/accents
LANGUAGES = {
    'en': 'English (US)',
    'en-uk': 'English (UK)',
    'en-au': 'English (Australia)',
    'en-in': 'English (India)',
    'hi': 'Hindi',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh-CN': 'Chinese (Simplified)',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ar': 'Arabic'
}

# Cache for cleanup - stores (filepath, timestamp)
audio_files_cache = {}
CLEANUP_INTERVAL = 300  # 5 minutes
MAX_FILE_AGE = 600  # 10 minutes


def cleanup_old_files():
    """Remove audio files older than MAX_FILE_AGE seconds."""
    current_time = time.time()
    files_to_remove = []
    
    for filepath, timestamp in list(audio_files_cache.items()):
        if current_time - timestamp > MAX_FILE_AGE:
            files_to_remove.append(filepath)
    
    for filepath in files_to_remove:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            del audio_files_cache[filepath]
        except Exception:
            pass


@audio_bp.route('/tools/text-to-speech')
def text_to_speech_page():
    """Render the Voice Studio page."""
    return render_template('tools/text_to_speech.html', languages=LANGUAGES)


@audio_bp.route('/api/text-to-speech', methods=['POST'])
def generate_speech():
    """
    Generate speech audio from text using gTTS.
    
    Expected JSON body:
    {
        "text": "Hello world",
        "language": "en",
        "slow": false
    }
    """
    try:
        # Import here to avoid startup issues if gtts not installed
        from gtts import gTTS
        
        data = request.get_json()
        
        text = data.get('text', '').strip()
        language = data.get('language', 'en')
        slow = data.get('slow', False)
        
        # Validate text
        if not text:
            return jsonify({'error': 'Please enter some text to convert to speech.'}), 400
        
        if len(text) > 1000:
            return jsonify({'error': 'Text is too long. Please limit to 1000 characters.'}), 400
        
        # Validate language
        lang_code = language.split('-')[0] if '-' in language else language
        tld = None
        
        # Handle regional variants
        if language == 'en-uk':
            lang_code = 'en'
            tld = 'co.uk'
        elif language == 'en-au':
            lang_code = 'en'
            tld = 'com.au'
        elif language == 'en-in':
            lang_code = 'en'
            tld = 'co.in'
        
        # Generate unique filename
        filename = f"speech_{uuid.uuid4().hex[:8]}.mp3"
        upload_folder = current_app.config.get('UPLOAD_FOLDER', 'static/temp')
        filepath = os.path.join(upload_folder, filename)
        
        # Ensure directory exists
        os.makedirs(upload_folder, exist_ok=True)
        
        # Generate audio
        if tld:
            tts = gTTS(text=text, lang=lang_code, slow=slow, tld=tld)
        else:
            tts = gTTS(text=text, lang=lang_code, slow=slow)
        
        tts.save(filepath)
        
        # Track file for cleanup
        audio_files_cache[filepath] = time.time()
        
        # Cleanup old files in background
        threading.Thread(target=cleanup_old_files, daemon=True).start()
        
        # Return URL to audio file
        audio_url = f'/static/temp/{filename}'
        
        return jsonify({
            'success': True,
            'audio_url': audio_url,
            'filename': filename
        })
        
    except Exception as e:
        error_msg = str(e)
        if 'Language not supported' in error_msg:
            return jsonify({'error': 'The selected language is not supported.'}), 400
        return jsonify({'error': f'Failed to generate audio: {error_msg}'}), 500


@audio_bp.route('/api/text-to-speech/download/<filename>')
def download_audio(filename):
    """Download the generated audio file."""
    try:
        # Security: only allow downloads from temp folder
        if not filename.startswith('speech_') or not filename.endswith('.mp3'):
            return jsonify({'error': 'Invalid file'}), 400
        
        upload_folder = current_app.config.get('UPLOAD_FOLDER', 'static/temp')
        filepath = os.path.join(upload_folder, filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found. It may have expired.'}), 404
        
        return send_file(filepath, as_attachment=True, download_name='speech.mp3')
        
    except Exception as e:
        return jsonify({'error': f'Download failed: {str(e)}'}), 500
