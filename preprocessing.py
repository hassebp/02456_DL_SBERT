# preprocessing.py
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

def preprocess_text(text):
    ps = PorterStemmer()
    words = [word for word in text.split() if word.lower() not in ENGLISH_STOP_WORDS]
    tokens = word_tokenize(" ".join(words))
    stemmed = [ps.stem(token) for token in tokens]
    cleaned_text = re.sub(r'[^\w\s]|_|\d', ' ', ' '.join(stemmed))
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text
