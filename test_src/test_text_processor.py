import re
class text_processor:
    #popular english stop words
    stop_words = {
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'while', 'with', 'without',
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did',
        'in', 'on', 'at', 'by', 'for', 'to', 'from',
        'of', 'as', 'that', 'this', 'these', 'those',
        # Add more stop words as needed
    }
    def __init__(self, remove_numbers=True, remove_stopwords=False, stemmer=None):
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.stemmer = stemmer
    
    def normalize_text_input(self, text):
        if isinstance(text, bytes):
            return text.decode('utf-8', errors='ignore')
        if isinstance(text, str):
            text = text.strip()
            text = re.sub(r"^b\s*(['\"])", r"\1", text)
            text = re.sub(r"^b\s+", "", text)
            if text.startswith((b"'", b'"')) and text.endswith(("'", '"')):
                return text[2:-1]
        return text

    def clean(self, text):
        if text is None:
            return ''
        text = re.sub(r'<.*?>', ' ', text)
        
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)

        text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)

        text = re.sub(r'\S+@\S+', ' ', text)
    
        text = text.lower()
        
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def remove_stopwords(self, tokens):
        if self.remove_stopwords:
            return [token for token in tokens if token not in self.stop_words]
        return tokens
        
    def process(self,text):
        text = self.normalize_text_input(text)
        text = self.clean(text)
        tokens = text.split()
        tokens = self.remove_stopwords(tokens)
        if self.stemmer is not None:
            tokens = [self.stemmer(token) for token in tokens]
        tokens = [token for token in tokens if token]  # Remove empty tokens
        return tokens