
"""
Text Sense Lab routes - NLP analysis using TextBlob and TheFuzz.
"""

from flask import Blueprint, render_template, request, jsonify

text_bp = Blueprint('text', __name__)

# Reference phrases for fuzzy matching
REFERENCE_PHRASES = [
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "neural network",
    "natural language processing",
    "computer vision",
    "data science",
    "big data",
    "cloud computing",
    "internet of things",
    "blockchain technology",
    "virtual reality",
    "augmented reality",
    "robotics automation",
    "quantum computing"
]


def get_sentiment_label(polarity: float) -> tuple:
    """Get sentiment label and emoji based on polarity score."""
    if polarity > 0.3:
        return 'Positive', '😊', '#22c55e'
    elif polarity > 0.1:
        return 'Slightly Positive', '🙂', '#84cc16'
    elif polarity < -0.3:
        return 'Negative', '😢', '#ef4444'
    elif polarity < -0.1:
        return 'Slightly Negative', '😕', '#f97316'
    else:
        return 'Neutral', '😐', '#6b7280'


def get_subjectivity_label(subjectivity: float) -> str:
    """Get subjectivity description based on score."""
    if subjectivity > 0.7:
        return "Highly subjective (opinion-based)"
    elif subjectivity > 0.4:
        return "Moderately subjective"
    else:
        return "Mostly objective (fact-based)"


@text_bp.route('/tools/text-sense')
def text_sense_page():
    """Render the Text Sense Lab page."""
    return render_template('tools/text_sense.html', reference_phrases=REFERENCE_PHRASES)


@text_bp.route('/api/text-analysis', methods=['POST'])
def analyze_text():
    """
    Analyze text for sentiment, spelling, and fuzzy matching.
    
    Expected JSON body:
    {
        "text": "Your text to analyze here"
    }
    """
    try:
        from textblob import TextBlob
        from thefuzz import fuzz, process
        
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Please enter some text to analyze.'}), 400
        
        if len(text) > 5000:
            return jsonify({'error': 'Text too long. Maximum is 5000 characters.'}), 400
        
        # Create TextBlob
        blob = TextBlob(text)
        
        # --- Sentiment Analysis ---
        polarity = blob.sentiment.polarity  # -1 to 1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1
        
        label, emoji, color = get_sentiment_label(polarity)
        subj_label = get_subjectivity_label(subjectivity)
        
        sentiment = {
            'polarity': round(polarity, 3),
            'polarity_percent': int((polarity + 1) * 50),  # Convert to 0-100 scale
            'subjectivity': round(subjectivity, 3),
            'subjectivity_percent': int(subjectivity * 100),
            'label': label,
            'emoji': emoji,
            'color': color,
            'subjectivity_label': subj_label
        }
        
        # --- Spelling Correction ---
        corrected = str(blob.correct())
        has_corrections = corrected.lower() != text.lower()
        
        spelling = {
            'original': text,
            'corrected': corrected,
            'has_corrections': has_corrections
        }
        
        # --- Fuzzy Matching ---
        # Find best matches from reference phrases
        matches = process.extract(text.lower(), REFERENCE_PHRASES, limit=5, scorer=fuzz.token_set_ratio)
        
        # Format matches with descriptions
        fuzzy_matches = []
        for phrase, score in matches:
            if score >= 30:  # Only show reasonably relevant matches
                fuzzy_matches.append({
                    'phrase': phrase.title(),
                    'score': score,
                    'quality': 'Excellent match' if score >= 80 else 
                              'Good match' if score >= 60 else 
                              'Partial match' if score >= 40 else 'Weak match'
                })
        
        # --- Additional Stats ---
        words = blob.words
        sentences = blob.sentences
        
        stats = {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'character_count': len(text),
            'avg_word_length': round(sum(len(w) for w in words) / max(len(words), 1), 1)
        }
        
        return jsonify({
            'success': True,
            'sentiment': sentiment,
            'spelling': spelling,
            'fuzzy_matches': fuzzy_matches,
            'stats': stats
        })
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@text_bp.route('/api/text-advanced', methods=['POST'])
def advanced_text_analysis():
    """
    Advanced NLP analysis including NER, Topic Modeling, Keywords, 
    Summarization, POS Tagging, Readability, and Subjectivity.
    """
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Please enter some text to analyze.'}), 400
        
        if len(text) > 10000:
            return jsonify({'error': 'Text too long. Maximum is 10000 characters.'}), 400
        
        results = {}
        
        # --- Named Entity Recognition (spaCy) ---
        try:
            import spacy
            nlp = spacy.load('en_core_web_sm')
            doc = nlp(text)
            
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_) or ent.label_
                })
            
            # Group by entity type
            entity_groups = {}
            for ent in entities:
                label = ent['label']
                if label not in entity_groups:
                    entity_groups[label] = []
                if ent['text'] not in [e['text'] for e in entity_groups[label]]:
                    entity_groups[label].append(ent)
            
            results['ner'] = {
                'entities': entities[:20],  # Limit to 20
                'groups': entity_groups,
                'total': len(entities)
            }
        except Exception as e:
            results['ner'] = {'error': str(e)}
        
        # --- POS Tagging (spaCy) ---
        try:
            import spacy
            if 'nlp' not in dir() or nlp is None:
                nlp = spacy.load('en_core_web_sm')
                doc = nlp(text)
            
            pos_tags = []
            for token in doc[:50]:  # Limit to 50 tokens
                if not token.is_space:
                    pos_tags.append({
                        'word': token.text,
                        'pos': token.pos_,
                        'tag': token.tag_,
                        'description': spacy.explain(token.tag_) or token.tag_
                    })
            
            # Count by POS type
            pos_counts = {}
            for token in doc:
                if not token.is_space:
                    pos = token.pos_
                    pos_counts[pos] = pos_counts.get(pos, 0) + 1
            
            results['pos'] = {
                'tags': pos_tags,
                'counts': pos_counts
            }
        except Exception as e:
            results['pos'] = {'error': str(e)}
        
        # --- Keyword Extraction (YAKE) ---
        try:
            import yake
            
            kw_extractor = yake.KeywordExtractor(
                lan="en",
                n=2,  # n-gram size
                dedupLim=0.7,
                top=10,
                features=None
            )
            keywords = kw_extractor.extract_keywords(text)
            
            results['keywords'] = {
                'keywords': [{'keyword': kw, 'score': round(1 - score, 3)} for kw, score in keywords]
            }
        except Exception as e:
            results['keywords'] = {'error': str(e)}
        
        # --- Readability Scoring (textstat) ---
        try:
            import textstat
            
            results['readability'] = {
                'flesch_reading_ease': round(textstat.flesch_reading_ease(text), 1),
                'flesch_kincaid_grade': round(textstat.flesch_kincaid_grade(text), 1),
                'smog_index': round(textstat.smog_index(text), 1),
                'coleman_liau_index': round(textstat.coleman_liau_index(text), 1),
                'automated_readability_index': round(textstat.automated_readability_index(text), 1),
                'dale_chall_readability_score': round(textstat.dale_chall_readability_score(text), 1),
                'reading_time_seconds': textstat.reading_time(text, ms_per_char=14.69),
                'text_standard': textstat.text_standard(text, float_output=False)
            }
            
            # Add interpretation
            fre = results['readability']['flesch_reading_ease']
            if fre >= 90:
                results['readability']['interpretation'] = "Very Easy - 5th grade level"
            elif fre >= 80:
                results['readability']['interpretation'] = "Easy - 6th grade level"
            elif fre >= 70:
                results['readability']['interpretation'] = "Fairly Easy - 7th grade level"
            elif fre >= 60:
                results['readability']['interpretation'] = "Standard - 8th-9th grade level"
            elif fre >= 50:
                results['readability']['interpretation'] = "Fairly Difficult - 10th-12th grade level"
            elif fre >= 30:
                results['readability']['interpretation'] = "Difficult - College level"
            else:
                results['readability']['interpretation'] = "Very Difficult - College graduate level"
                
        except Exception as e:
            results['readability'] = {'error': str(e)}
        
        # --- Subjectivity Analysis (TextBlob) ---
        try:
            from textblob import TextBlob
            
            blob = TextBlob(text)
            subjectivity = blob.sentiment.subjectivity
            polarity = blob.sentiment.polarity
            
            # Analyze sentences for subjectivity
            subjective_sentences = []
            objective_sentences = []
            for sentence in blob.sentences:
                if sentence.sentiment.subjectivity > 0.5:
                    subjective_sentences.append(str(sentence))
                else:
                    objective_sentences.append(str(sentence))
            
            results['subjectivity'] = {
                'score': round(subjectivity, 3),
                'percent': int(subjectivity * 100),
                'polarity': round(polarity, 3),
                'label': 'Opinion-based' if subjectivity > 0.5 else 'Fact-based',
                'subjective_sentences': subjective_sentences[:5],
                'objective_sentences': objective_sentences[:5]
            }
        except Exception as e:
            results['subjectivity'] = {'error': str(e)}
        
        # --- Text Summarization (Sumy) ---
        try:
            from sumy.parsers.plaintext import PlaintextParser
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.summarizers.lsa import LsaSummarizer
            from sumy.nlp.stemmers import Stemmer
            from sumy.utils import get_stop_words
            
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            stemmer = Stemmer("english")
            summarizer = LsaSummarizer(stemmer)
            summarizer.stop_words = get_stop_words("english")
            
            # Get 3 sentences or fewer if text is short
            sentence_count = min(3, len(list(parser.document.sentences)))
            summary_sentences = summarizer(parser.document, sentence_count)
            
            summary = ' '.join([str(sentence) for sentence in summary_sentences])
            
            results['summary'] = {
                'text': summary,
                'sentence_count': sentence_count,
                'compression_ratio': round(len(summary) / len(text) * 100, 1) if text else 0
            }
        except Exception as e:
            results['summary'] = {'error': str(e)}
        
        # --- Topic Modeling (Simple keyword-based for speed) ---
        try:
            # Use a simpler approach for real-time analysis
            from collections import Counter
            import re
            
            # Extract meaningful words
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            
            # Common stop words
            stop_words = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'will', 
                         'would', 'could', 'should', 'there', 'their', 'they', 'them',
                         'what', 'when', 'where', 'which', 'while', 'about', 'after',
                         'before', 'being', 'between', 'both', 'each', 'either', 'more',
                         'most', 'other', 'some', 'such', 'than', 'these', 'those', 'very'}
            
            filtered_words = [w for w in words if w not in stop_words]
            word_freq = Counter(filtered_words)
            
            # Get top topics (most frequent meaningful words)
            top_topics = word_freq.most_common(8)
            
            results['topics'] = {
                'topics': [{'topic': word, 'frequency': freq} for word, freq in top_topics],
                'method': 'Frequency-based extraction'
            }
        except Exception as e:
            results['topics'] = {'error': str(e)}
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': f'Advanced analysis failed: {str(e)}'}), 500
