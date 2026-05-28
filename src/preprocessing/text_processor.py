\
\
\
\
\
\
\
\
\
   

import re


class TextProcessor:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       
    
                              
    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'am', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
        'who', 'when', 'where', 'why', 'how'
    }
    
    def __init__(self, remove_stopwords=False, remove_numbers=True, stemmer=None):
\
\
\
\
\
\
\
\
\
           
                                                                        
        self.remove_stopwords = remove_stopwords
        self.remove_numbers = remove_numbers
        self.stemmer = stemmer

    def _normalize_text_input(self, text):
        if isinstance(text, bytes):
            return text.decode('utf-8', errors='ignore')
        if isinstance(text, str):
            text = text.strip()
            text = re.sub(r"^b\s*(['\"])", r"\1", text)
            text = re.sub(r"^b\s+", "", text)
            if (text.startswith("b'") and text.endswith("'")) or (text.startswith('b"') and text.endswith('"')):
                return text[2:-1]
        return text

    def clean(self, text):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        if text is None:
            return ''

        text = self._normalize_text_input(text)

        if not isinstance(text, str):
            text = str(text)

                          
        text = re.sub(r'<.*?>', '', text)
        
                              
        text = text.lower()
        
                     
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
                                
        text = re.sub(r'\S+@\S+', '', text)
        
                                     
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
                                                                                     
        text = re.sub(r"[^\w\s']", ' ', text, flags=re.UNICODE)
                                                        
        text = re.sub(r'_+', ' ', text)
        
                                 
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        return text.split()
    
    def remove_stopwords_tokens(self, tokens):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        return [token for token in tokens if token.lower() not in self.STOPWORDS]
    
    def process(self, text):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
                    
        text = self.clean(text)
        
                  
        tokens = self.tokenize(text)
        
                                                                       
        if self.remove_stopwords:
            tokens = self.remove_stopwords_tokens(tokens)
        
                                              
        if self.stemmer is not None:
            tokens = [self.stemmer(token) for token in tokens]

                             
        tokens = [token for token in tokens if token]
        
        return tokens
    
    def __call__(self, text):
                                                               
        return self.process(text)
    
    def __repr__(self):
                                    
        return f"TextProcessor(remove_stopwords={self.remove_stopwords}, remove_numbers={self.remove_numbers})"
