"""
Home and About page routes for NeuralPlayground.
"""

from flask import Blueprint, render_template

home_bp = Blueprint('home', __name__)


@home_bp.route('/')
def index():
    """
    Home page displaying the hero section and tool cards grid.
    """
    tools = [
        {
            'id': 'fake-data',
            'name': 'Fake Data Lab',
            'icon': '🎲',
            'description': 'Generate realistic fake data for testing – names, emails, addresses, and more.',
            'url': '/tools/fake-data',
            'popular': False
        },
        {
            'id': 'text-to-speech',
            'name': 'Voice Studio',
            'icon': '🔊',
            'description': 'Turn your text into natural-sounding speech in multiple languages.',
            'url': '/tools/text-to-speech',
            'popular': False
        },
        {
            'id': 'ascii-art',
            'name': 'ASCII Art Lab',
            'icon': '🎨',
            'description': 'Transform any image into creative text-based ASCII art.',
            'url': '/tools/ascii-art',
            'popular': False
        },
        {
            'id': 'qr-code',
            'name': 'QR & Codes Lab',
            'icon': '📱',
            'description': 'Create QR codes for websites, text, or contact information instantly.',
            'url': '/tools/qr-code',
            'popular': False
        },
        {
            'id': 'text-sense',
            'name': 'Text Sense Lab',
            'icon': '📝',
            'description': 'Analyze text for sentiment, fix spelling, and find similar phrases.',
            'url': '/tools/text-sense',
            'popular': True
        },
        {
            'id': 'vision-playground',
            'name': 'Vision Playground',
            'icon': '👁️',
            'description': 'Detect faces and hands in photos using computer vision.',
            'url': '/tools/vision-playground',
            'popular': True
        },
        {
            'id': 'emotion-mirror',
            'name': 'Emotion Mirror',
            'icon': '😊',
            'description': 'Discover what emotions a photo might express using AI.',
            'url': '/tools/emotion-mirror',
            'popular': True
        }
    ]
    
    popular_tools = [t for t in tools if t.get('popular')][:3]
    
    return render_template('index.html', tools=tools, popular_tools=popular_tools)


@home_bp.route('/about')
def about():
    """
    About page with information about NeuralPlayground and the libraries used.
    """
    libraries = [
        {
            'name': 'Faker',
            'description': 'Generates realistic fake data like names, addresses, and phone numbers.',
            'used_in': 'Fake Data Lab'
        },
        {
            'name': 'gTTS (Google Text-to-Speech)',
            'description': 'Converts text into natural-sounding speech audio.',
            'used_in': 'Voice Studio'
        },
        {
            'name': 'OpenCV',
            'description': 'Industry-standard library for image and video processing.',
            'used_in': 'ASCII Art Lab, Vision Playground, Emotion Mirror'
        },
        {
            'name': 'MediaPipe',
            'description': 'Google\'s ML framework for real-time face and hand detection.',
            'used_in': 'Vision Playground'
        },
        {
            'name': 'PyQRCode',
            'description': 'Creates QR codes from text or URLs.',
            'used_in': 'QR & Codes Lab'
        },
        {
            'name': 'TextBlob',
            'description': 'Analyzes text for sentiment and provides spelling correction.',
            'used_in': 'Text Sense Lab'
        },
        {
            'name': 'TheFuzz',
            'description': 'Finds similar text even with typos using fuzzy matching.',
            'used_in': 'Text Sense Lab'
        },
        {
            'name': 'FER (Facial Expression Recognition)',
            'description': 'Detects emotions in facial expressions using deep learning with TensorFlow.',
            'used_in': 'Emotion Mirror'
        },
        {
            'name': 'pyfiglet',
            'description': 'Converts text into ASCII art with various font styles.',
            'used_in': 'ASCII Art Lab (Text to ASCII)'
        }
    ]
    
    return render_template('about.html', libraries=libraries)
