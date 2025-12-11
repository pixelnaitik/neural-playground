"""
Vision routes - Face and Hand detection using MediaPipe.
OPTIMIZED for accuracy and speed.

Key Optimizations:
- Singleton patterns for heavy ML models (MediaPipe, FER).
- Top-level imports for core libraries (cv2, numpy).
- Base64 decoding helper to reduce code duplication.
- Type hinting and robust error handling.
"""

import base64
import cv2
import numpy as np
from flask import Blueprint, render_template, request, jsonify
from typing import List, Dict, Any, Optional, Tuple, Union

vision_bp = Blueprint('vision', __name__)

# ========== SINGLETON DETECTORS ==========
_face_detector = None
_hand_detector = None
_emotion_detector_fast = None  # mtcnn=False (for live)
_emotion_detector_accurate = None  # mtcnn=True (for static)

def get_mediapipe_face():
    """Get singleton MediaPipe face detector."""
    global _face_detector
    if _face_detector is None:
        import mediapipe as mp
        _face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # 1 = full range model (better accuracy)
            min_detection_confidence=0.7  # High confidence to reduce false positives
        )
    return _face_detector

def get_mediapipe_hands():
    """Get singleton MediaPipe hands detector."""
    global _hand_detector
    if _hand_detector is None:
        import mediapipe as mp
        _hand_detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
    return _hand_detector

def get_emotion_detector(mtcnn: bool = False):
    """
    Get singleton FER emotion detector.
    Args:
        mtcnn: If True, uses MTCNN (slower, more accurate). If False, uses OpenCV Haar (faster).
    """
    global _emotion_detector_fast, _emotion_detector_accurate
    
    if mtcnn:
        if _emotion_detector_accurate is None:
            from fer.fer import FER
            _emotion_detector_accurate = FER(mtcnn=True)
        return _emotion_detector_accurate
    else:
        if _emotion_detector_fast is None:
            from fer.fer import FER
            _emotion_detector_fast = FER(mtcnn=False)
        return _emotion_detector_fast

# ========== HELPER FUNCTIONS ==========

def decode_image(image_data: bytes) -> np.ndarray:
    """Decode image bytes to OpenCV format."""
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img

def decode_base64_frame(frame_data: str) -> np.ndarray:
    """
    Decode base64 frame string from frontend.
    Removes data URL prefix if present.
    """
    if ',' in frame_data:
        frame_data = frame_data.split(',')[1]
    
    img_bytes = base64.b64decode(frame_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not decode base64 frame")
    return img

def encode_image_fast(img: np.ndarray, quality: int = 80) -> str:
    """Encode OpenCV image to JPEG base64 string."""
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode('utf-8')

# ========== ROUTES ==========

@vision_bp.route('/tools/vision-playground')
def vision_playground_page():
    return render_template('tools/vision_playground.html')

@vision_bp.route('/tools/emotion-mirror')
def emotion_mirror_page():
    return render_template('tools/emotion_mirror.html')

@vision_bp.route('/api/vision/faces', methods=['POST'])
def detect_faces():
    """Detect faces in uploaded static image."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Please upload an image file.'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected.'}), 400
        
        img = decode_image(file.read())
        original_img = img.copy()
        h, w = img.shape[:2]
        
        # Use MediaPipe face detection
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_detector = get_mediapipe_face()
        results = face_detector.process(rgb)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                # Skip low confidence detections
                if detection.score[0] < 0.7:
                    continue
                    
                bbox = detection.location_data.relative_bounding_box
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                
                # Skip very small detections (likely false positives)
                if bw < 40 or bh < 40:
                    continue
                
                cv2.rectangle(img, (x, y), (x+bw, y+bh), (255, 100, 50), 3)
                cv2.putText(img, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 50), 2)
                faces.append({'x': x, 'y': y, 'width': bw, 'height': bh})
        
        return jsonify({
            'success': True,
            'faces_detected': len(faces),
            'original_image': f'data:image/jpeg;base64,{encode_image_fast(original_img)}',
            'processed_image': f'data:image/jpeg;base64,{encode_image_fast(img)}',
            'face_locations': faces
        })
        
    except Exception as e:
        return jsonify({'error': f'Face detection failed: {str(e)}'}), 500


@vision_bp.route('/api/vision/hands', methods=['POST'])
def detect_hands():
    """Detect hands in uploaded static image."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Please upload an image file.'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected.'}), 400
        
        img = decode_image(file.read())
        original_img = img.copy()
        
        # Helper: lazy import mp drawing utils only when needed
        import mediapipe as mp
        mp_draw = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hand_detector = get_mediapipe_hands()
        results = hand_detector.process(rgb)
        
        hands_detected = 0
        hand_info = []
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hands_detected += 1
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                handedness = 'Unknown'
                if results.multi_handedness:
                    handedness = results.multi_handedness[idx].classification[0].label
                
                hand_info.append({'hand_type': handedness})
        
        return jsonify({
            'success': True,
            'hands_detected': hands_detected,
            'original_image': f'data:image/jpeg;base64,{encode_image_fast(original_img)}',
            'processed_image': f'data:image/jpeg;base64,{encode_image_fast(img)}',
            'hand_info': hand_info
        })
        
    except Exception as e:
        return jsonify({'error': f'Hand detection failed: {str(e)}'}), 500


@vision_bp.route('/api/emotion-detect', methods=['POST'])
def detect_emotions():
    """Detect emotions in uploaded static image (High Accuracy Mode)."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Please upload an image file.'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected.'}), 400
        
        img = decode_image(file.read())
        original_img = img.copy()
        emotions_data = []
        
        try:
            # Use MTCNN=True for static images (better accuracy)
            detector = get_emotion_detector(mtcnn=True)
            results = detector.detect_emotions(img)
            
            emotion_emojis = {'angry': '😠', 'disgust': '🤢', 'fear': '😨', 'happy': '😊', 'sad': '😢', 'surprise': '😲', 'neutral': '😐'}
            
            for i, result in enumerate(results):
                box = result.get('box', [0, 0, 0, 0])
                x, y, w, h = box
                
                # FILTER: Skip very small faces
                if w < 50 or h < 50:
                    continue
                    
                emotions = result.get('emotions', {})
                
                if emotions:
                    # CALIBRATION LOGIC
                    calibrated = emotions.copy()
                    
                    if calibrated.get('fear', 0) > 0 and calibrated.get('neutral', 0) > 0.15:
                        calibrated['fear'] = calibrated['fear'] * 0.5
                    
                    max_emotion = max(calibrated.values())
                    if max_emotion < 0.4:
                        calibrated['neutral'] = calibrated.get('neutral', 0) + 0.2
                    
                    dominant = max(calibrated, key=calibrated.get)
                    confidence = emotions.get(dominant, 0)
                    
                    if confidence < 0.25:
                        dominant = 'neutral'
                        confidence = emotions.get('neutral', 0.5)
                else:
                    continue
                
                cv2.rectangle(img, (x, y), (x+w, y+h), (50, 200, 100), 3)
                cv2.putText(img, f"{dominant}: {int(confidence*100)}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 200, 100), 2)
                
                emotions_data.append({
                    'face_id': i + 1,
                    'dominant_emotion': dominant,
                    'dominant_emoji': emotion_emojis.get(dominant, '🤔'),
                    'confidence': round(confidence * 100, 1),
                    'all_emotions': {k: round(v * 100, 1) for k, v in emotions.items()}
                })
        except ImportError:
            pass  # FER might not be installed or configured
        
        return jsonify({
            'success': True,
            'faces_detected': len(emotions_data),
            'original_image': f'data:image/jpeg;base64,{encode_image_fast(original_img)}',
            'processed_image': f'data:image/jpeg;base64,{encode_image_fast(img)}',
            'emotions': emotions_data
        })
        
    except Exception as e:
        return jsonify({'error': f'Emotion detection failed: {str(e)}'}), 500


# ========== LIVE STREAMING - HIGH PERFORMANCE ==========

@vision_bp.route('/api/vision/live/all', methods=['POST'])
def live_detect_all():
    """
    Live face + hand detection. 
    Returns coordinates only for frontend rendering (Max FPS).
    """
    try:
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({'error': 'No frame'}), 400
        
        try:
            img = decode_base64_frame(data['frame'])
        except ValueError:
            return jsonify({'error': 'Bad frame'}), 400
        
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        faces = []
        hands = []
        
        # Face detection
        face_detector = get_mediapipe_face()
        face_results = face_detector.process(rgb)
        
        if face_results.detections:
            for detection in face_results.detections:
                if detection.score[0] < 0.7:
                    continue
                    
                bbox = detection.location_data.relative_bounding_box
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                
                if bw < 30 or bh < 30:
                    continue
                    
                faces.append({'x': x, 'y': y, 'w': bw, 'h': bh})
        
        # Hand detection
        hand_detector = get_mediapipe_hands()
        hand_results = hand_detector.process(rgb)
        
        if hand_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                # Calculate bounding box
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                
                x_min = max(0, int(min(x_coords)) - 20)
                y_min = max(0, int(min(y_coords)) - 20)
                x_max = min(w, int(max(x_coords)) + 20)
                y_max = min(h, int(max(y_coords)) + 20)
                
                handedness = 'Hand'
                if hand_results.multi_handedness:
                    handedness = hand_results.multi_handedness[idx].classification[0].label
                
                hands.append({
                    'x': x_min, 'y': y_min,
                    'w': x_max - x_min, 'h': y_max - y_min,
                    'type': handedness
                })
        
        return jsonify({
            'success': True,
            'faces': faces, 'hands': hands,
            'faces_detected': len(faces), 'hands_detected': len(hands)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@vision_bp.route('/api/vision/live/faces', methods=['POST'])
def live_detect_faces():
    """Fast live face detection."""
    try:
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({'error': 'No frame'}), 400
        
        try:
            img = decode_base64_frame(data['frame'])
        except ValueError:
            return jsonify({'error': 'Bad frame'}), 400
        
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        face_detector = get_mediapipe_face()
        results = face_detector.process(rgb)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                if detection.score[0] < 0.7: continue
                bbox = detection.location_data.relative_bounding_box
                faces.append({
                    'x': max(0, int(bbox.xmin * w)),
                    'y': max(0, int(bbox.ymin * h)),
                    'w': int(bbox.width * w),
                    'h': int(bbox.height * h)
                })
        
        return jsonify({'success': True, 'faces': faces, 'faces_detected': len(faces)})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@vision_bp.route('/api/vision/live/hands', methods=['POST'])
def live_detect_hands():
    """Fast live hand detection."""
    try:
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({'error': 'No frame'}), 400
        
        try:
            img = decode_base64_frame(data['frame'])
        except ValueError:
            return jsonify({'error': 'Bad frame'}), 400
        
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        hand_detector = get_mediapipe_hands()
        results = hand_detector.process(rgb)
        
        hands = []
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                
                x_min = max(0, int(min(x_coords)) - 20)
                y_min = max(0, int(min(y_coords)) - 20)
                x_max = min(w, int(max(x_coords)) + 20)
                y_max = min(h, int(max(y_coords)) + 20)
                
                handedness = 'Hand'
                if results.multi_handedness:
                    handedness = results.multi_handedness[idx].classification[0].label
                
                hands.append({
                    'x': x_min, 'y': y_min,
                    'w': x_max - x_min, 'h': y_max - y_min,
                    'type': handedness
                })
        
        return jsonify({'success': True, 'hands': hands, 'hands_detected': len(hands)})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@vision_bp.route('/api/emotion/live', methods=['POST'])
def live_detect_emotion():
    """
    Live emotion detection.
    Optimized to reuse FER detector (singleton).
    """
    try:
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({'error': 'No frame'}), 400
        
        try:
            img = decode_base64_frame(data['frame'])
        except ValueError:
            return jsonify({'error': 'Bad frame'}), 400
        
        emotions_data = []
        
        try:
            # Singleton detector (mtcnn=False for speed)
            detector = get_emotion_detector(mtcnn=False)
            results = detector.detect_emotions(img)
            
            emotion_emojis = {'angry': '😠', 'disgust': '🤢', 'fear': '😨', 'happy': '😊', 'sad': '😢', 'surprise': '😲', 'neutral': '😐'}
            
            for i, result in enumerate(results):
                box = result.get('box', [0, 0, 0, 0])
                x, y, w, h = box
                
                if w < 40 or h < 40:
                    continue
                
                emotions = result.get('emotions', {})
                if not emotions: continue
                
                # CALIBRATION
                calibrated = emotions.copy()
                if calibrated.get('fear', 0) > 0 and calibrated.get('neutral', 0) > 0.15:
                    calibrated['fear'] = calibrated['fear'] * 0.5
                
                max_emotion = max(calibrated.values())
                if max_emotion < 0.4:
                    calibrated['neutral'] = calibrated.get('neutral', 0) + 0.2
                
                dominant = max(calibrated, key=calibrated.get)
                confidence = emotions.get(dominant, 0)
                
                if confidence < 0.25:
                    dominant = 'neutral'
                    confidence = emotions.get('neutral', 0.5)
                
                cv2.rectangle(img, (x, y), (x+w, y+h), (50, 200, 100), 3)
                cv2.putText(img, f"{dominant}: {int(confidence*100)}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 200, 100), 2)
                
                emotions_data.append({
                    'face_id': i + 1,
                    'dominant_emotion': dominant,
                    'dominant_emoji': emotion_emojis.get(dominant, '🤔'),
                    'confidence': round(confidence * 100, 1),
                    'all_emotions': {k: round(v * 100, 1) for k, v in emotions.items()}
                })
        except ImportError:
            pass
        except Exception as e:
            print(f"Emotion detection error: {e}") # Non-blocking error logging
        
        return jsonify({
            'success': True,
            'faces_detected': len(emotions_data),
            'processed_frame': f'data:image/jpeg;base64,{encode_image_fast(img)}',
            'emotions': emotions_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
