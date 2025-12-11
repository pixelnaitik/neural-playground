"""
NeuralPlayground - Main Flask Application

A web application providing interactive AI/ML tools for non-technical users.
"""

import os
from flask import Flask

# Import route blueprints
from routes.routes_home import home_bp
from routes.routes_data import data_bp
from routes.routes_audio import audio_bp
from routes.routes_image import image_bp
from routes.routes_text import text_bp
from routes.routes_vision import vision_bp

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Configuration
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, 'temp')
    
    # Ensure temp directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Register blueprints
    app.register_blueprint(home_bp)
    app.register_blueprint(data_bp)
    app.register_blueprint(audio_bp)
    app.register_blueprint(image_bp)
    app.register_blueprint(text_bp)
    app.register_blueprint(vision_bp)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
