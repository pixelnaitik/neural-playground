# NeuralPlayground

An interactive web application that makes AI and machine learning tools accessible to everyone. Explore text analysis, image processing, audio generation, and computer vision – all in your browser, no coding required.

![NeuralPlayground](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)

## Features

### 🎲 Fake Data Lab
Generate realistic synthetic data for testing – names, emails, addresses, and more. Supports multiple locales and custom schemas.

### 🔊 Voice Studio
Convert text to natural-sounding speech in 15+ languages using Google Text-to-Speech.

### 🎨 ASCII Art Lab
Transform any image into creative text-based ASCII art using OpenCV image processing.

### 📱 QR & Codes Lab
Create QR codes instantly for websites, text, or contact information. Download as PNG or SVG.

### 📝 Text Sense Lab
Analyze text sentiment, fix spelling mistakes, and find similar phrases using TextBlob and fuzzy matching.

### 👁️ Vision Playground
Detect faces and hands in photos using computer vision (OpenCV + Cvzone).

### 😊 Emotion Mirror
Discover what emotions are expressed in photos using facial expression recognition (FER).

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. Clone or download this repository:
```bash
cd neuralplayground
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download TextBlob corpora (required for text analysis):
```bash
python -m textblob.download_corpora
```

## Running the Application

Start the Flask development server:

```bash
python app.py
```

The application will be available at: **http://localhost:5000**

## Project Structure

```
neuralplayground/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── routes/
│   ├── routes_home.py        # Home, About pages
│   ├── routes_data.py        # Fake Data Lab
│   ├── routes_audio.py       # Voice Studio
│   ├── routes_image.py       # ASCII Art, QR Code
│   ├── routes_text.py        # Text Sense Lab
│   └── routes_vision.py      # Vision, Emotion detection
├── static/
│   ├── css/style.css         # Global styles
│   ├── js/app.js             # Client-side utilities
│   └── temp/                 # Temporary files (auto-cleaned)
└── templates/
    ├── base.html             # Base template
    ├── index.html            # Home page
    ├── about.html            # About page
    └── tools/                # Tool-specific pages
```

## Libraries Used

| Library | Purpose |
|---------|---------|
| **Flask** | Web framework |
| **Faker** | Synthetic data generation |
| **gTTS** | Google Text-to-Speech |
| **OpenCV** | Image processing |
| **PyQRCode** | QR code generation |
| **TextBlob** | NLP and sentiment analysis |
| **TheFuzz** | Fuzzy string matching |
| **Cvzone** | Hand/pose detection |
| **FER** | Facial emotion recognition |

## Configuration

The application uses sensible defaults. You can modify these in `app.py`:

- `MAX_CONTENT_LENGTH`: Maximum upload file size (default: 16MB)
- `UPLOAD_FOLDER`: Temporary file storage location

## Notes

- **Temporary files**: Audio and processed images are stored temporarily and auto-cleaned.
- **Privacy**: No data is stored permanently. All processing happens server-side.
- **AI Accuracy**: Results from ML models (emotion, sentiment) are estimates and may not be accurate.

## Troubleshooting

### "No module named X"
Run `pip install -r requirements.txt` to install all dependencies.

### TextBlob errors
Run `python -m textblob.download_corpora` to download required language data.

### OpenCV/FER issues on Windows
Some computers may need Visual C++ Redistributable. Download from Microsoft if you see DLL errors.

### Hand detection not working
Cvzone requires MediaPipe which may have platform-specific requirements. Face detection should work on all platforms.

## License

This project is for educational and experimental purposes.

---

Built with ❤️ for AI enthusiasts
