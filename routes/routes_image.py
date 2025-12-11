


"""
Image processing routes - ASCII Art Lab and QR Code Lab.
Uses OpenCV for image processing and PyQRCode for QR generation.
"""

import os
import io
import uuid
import base64
from flask import Blueprint, render_template, request, jsonify, current_app

image_bp = Blueprint('image', __name__)

# ASCII characters from dark to light
ASCII_CHARS = '@%#*+=-:. '

# Size presets for ASCII art
SIZE_PRESETS = {
    'tiny': {'width': 40, 'name': 'Tiny (40 chars wide)'},
    'small': {'width': 60, 'name': 'Small (60 chars wide)'},
    'medium': {'width': 80, 'name': 'Medium (80 chars wide)'},
    'large': {'width': 120, 'name': 'Large (120 chars wide)'}
}

# Error correction levels for QR codes
ERROR_CORRECTION = {
    'L': {'name': 'Low (7%)', 'description': 'Allows about 7% of the code to be damaged'},
    'M': {'name': 'Medium (15%)', 'description': 'Allows about 15% of the code to be damaged'},
    'Q': {'name': 'Quartile (25%)', 'description': 'Allows about 25% of the code to be damaged'},
    'H': {'name': 'High (30%)', 'description': 'Allows about 30% of the code to be damaged'}
}


def image_to_ascii(image_data: bytes, target_width: int = 80) -> tuple:
    """
    Convert image bytes to ASCII art.
    Returns tuple of (ascii_string, aesthetics_score or None).
    """
    import cv2
    import numpy as np
    
    # Decode image
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not decode image")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate aspect ratio and new height
    height, width = gray.shape
    aspect_ratio = height / width
    # Adjust for character aspect ratio (characters are taller than wide)
    new_height = int(target_width * aspect_ratio * 0.5)
    
    # Resize
    resized = cv2.resize(gray, (target_width, new_height))
    
    # Convert pixels to ASCII
    ascii_lines = []
    for row in resized:
        line = ''
        for pixel in row:
            # Map pixel value (0-255) to ASCII character
            char_index = int(pixel / 255 * (len(ASCII_CHARS) - 1))
            line += ASCII_CHARS[char_index]
        ascii_lines.append(line)
    
    ascii_art = '\n'.join(ascii_lines)
    
    # Simple "aesthetics score" placeholder
    # In reality, this would use a proper aesthetics model
    # Here we calculate a simple score based on contrast and detail
    contrast = np.std(gray) / 128  # Normalized standard deviation
    edges = cv2.Canny(gray, 100, 200)
    detail = np.mean(edges) / 255
    aesthetics_score = min(100, int((contrast * 50 + detail * 50)))
    
    return ascii_art, aesthetics_score


@image_bp.route('/tools/ascii-art')
def ascii_art_page():
    """Render the ASCII Art Lab page."""
    return render_template('tools/ascii_art.html', size_presets=SIZE_PRESETS)


@image_bp.route('/api/ascii-art', methods=['POST'])
def generate_ascii_art():
    """
    Convert uploaded image to ASCII art.
    
    Expects multipart form data with:
    - image: The image file
    - size: Size preset (tiny, small, medium, large)
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Please upload an image file.'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected.'}), 400
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
        ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
        if ext not in allowed_extensions:
            return jsonify({'error': 'Please upload a PNG, JPG, or GIF image.'}), 400
        
        # Check file size (max 5MB)
        file.seek(0, 2)  # Seek to end
        size = file.tell()
        file.seek(0)  # Seek back to start
        
        if size > 5 * 1024 * 1024:
            return jsonify({'error': 'Image too large. Maximum size is 5MB.'}), 400
        
        # Get size preset
        size_preset = request.form.get('size', 'medium')
        if size_preset not in SIZE_PRESETS:
            size_preset = 'medium'
        
        target_width = SIZE_PRESETS[size_preset]['width']
        
        # Convert to ASCII
        image_data = file.read()
        ascii_art, aesthetics_score = image_to_ascii(image_data, target_width)
        
        return jsonify({
            'success': True,
            'ascii_art': ascii_art,
            'aesthetics_score': aesthetics_score,
            'width': target_width,
            'lines': ascii_art.count('\n') + 1
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to convert image: {str(e)}'}), 500


@image_bp.route('/api/ascii-text', methods=['POST'])
def generate_ascii_text():
    """
    Convert text to ASCII art using pyfiglet.
    
    Expected JSON body:
    {
        "text": "Hello",
        "font": "standard"
    }
    """
    try:
        import pyfiglet
        
        data = request.get_json()
        text = data.get('text', '').strip()
        font = data.get('font', 'standard')
        
        if not text:
            return jsonify({'error': 'Please enter some text.'}), 400
        
        if len(text) > 50:
            return jsonify({'error': 'Text too long. Maximum is 50 characters.'}), 400
        
        # Get available fonts
        available_fonts = pyfiglet.FigletFont.getFonts()
        
        # Validate font
        if font not in available_fonts:
            font = 'standard'
        
        # Generate ASCII art from text
        fig = pyfiglet.Figlet(font=font)
        ascii_art = fig.renderText(text)
        
        return jsonify({
            'success': True,
            'ascii_art': ascii_art,
            'font': font,
            'available_fonts': sorted(available_fonts)[:30]  # Return first 30 fonts
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to generate ASCII text: {str(e)}'}), 500


@image_bp.route('/tools/qr-code')
def qr_code_page():
    """Render the QR Code Lab page."""
    return render_template('tools/qr_code.html', error_correction=ERROR_CORRECTION)


@image_bp.route('/api/qr-code', methods=['POST'])
def generate_qr_code():
    """
    Generate a QR code from text or URL.
    
    Expected JSON body:
    {
        "content": "https://example.com",
        "error_correction": "M"
    }
    """
    try:
        import pyqrcode
        import png
        
        data = request.get_json()
        
        content = data.get('content', '').strip()
        error_level = data.get('error_correction', 'M').upper()
        
        if not content:
            return jsonify({'error': 'Please enter text or a URL for the QR code.'}), 400
        
        if len(content) > 2000:
            return jsonify({'error': 'Content too long. Maximum is 2000 characters.'}), 400
        
        if error_level not in ERROR_CORRECTION:
            error_level = 'M'
        
        # Generate QR code
        qr = pyqrcode.create(content, error=error_level)
        
        # Generate PNG as base64
        buffer = io.BytesIO()
        qr.png(buffer, scale=8, module_color=[0, 0, 0, 255], background=[255, 255, 255, 255])
        buffer.seek(0)
        png_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        # Generate SVG
        svg_buffer = io.BytesIO()
        qr.svg(svg_buffer, scale=8)
        svg_buffer.seek(0)
        svg_content = svg_buffer.read().decode('utf-8')
        
        return jsonify({
            'success': True,
            'png_data_uri': f'data:image/png;base64,{png_base64}',
            'svg': svg_content,
            'content': content,
            'error_level': error_level
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to generate QR code: {str(e)}'}), 500
