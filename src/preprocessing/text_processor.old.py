"""
Text preprocessing utilities for natural language processing.

This module provides text cleaning and tokenization functions commonly used
in NLP pipelines before feature extraction and model training.

Classes:
--------
TextProcessor : Text cleaning, tokenization, and normalization
"""

import re
import string


class TextProcessor:
    """
    Text preprocessing and normalization utilities.
    
    This class provides methods for:
    - Removing HTML tags and special characters
    - Lowercasing and whitespace normalization
    - Tokenization (word splitting)
    - Stop word removal (optional)
    
    Methods:
    --------
    clean(text) : Clean text (remove HTML, special chars, lowercase)
    tokenize(text) : Split text into tokens
    process(text) : Full pipeline (clean + tokenize)
    remove_stopwords(tokens) : Remove common English stopwords
    
    Example:
    --------
    >>> processor = TextProcessor()
    >>> text = "<p>This is Great! 123</p>"
    >>> tokens = processor.process(text)
    >>> print(tokens)
    ['this', 'is', 'great']
    """
    
    # Common English stopwords
    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'am', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
        'who', 'when', 'where', 'why', 'how'
    }
    
    def __init__(self, remove_stopwords=False, remove_numbers=True):
        """
        Initialize TextProcessor.
        
        Parameters:
        -----------
        remove_stopwords : bool, default=False
            If True, remove common English stopwords
        remove_numbers : bool, default=True
            If True, remove numeric characters
        """
        # store flag separately from method name to avoid name collision
        self.remove_stopwords = remove_stopwords
        self.remove_numbers = remove_numbers

    def clean(self, text):
        """
        Clean text by removing HTML tags, special characters, etc.
        
        Operations:
        - Remove HTML tags: <...>
        - Remove special characters (keep only letters and spaces)
        - Convert to lowercase
        - Remove extra whitespace
        
        Parameters:
        -----------
        text : str
            Raw text to clean
        
        Returns:
        --------
        cleaned_text : str
            Cleaned text
        
        Example:
        --------
        >>> processor = TextProcessor()
        >>> text = "<p>Hello World! 123</p>"
        >>> processor.clean(text)
        'hello world'
        """
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers if requested
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove special characters (keep only letters, spaces, and apostrophes)
        text = re.sub(r"[^a-zA-Z\s']", '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text):
        """
        Tokenize text into words.
        
        Simple whitespace-based tokenization.
        
        Parameters:
        -----------
        text : str
            Text to tokenize
        
        Returns:
        --------
        tokens : list of str
            List of word tokens
        
        Example:
        --------
        >>> processor = TextProcessor()
        >>> processor.tokenize("hello world python")
        ['hello', 'world', 'python']
        """
        return text.split()
    
    def remove_stopwords_tokens(self, tokens):
        """
        Remove stopwords from token list.
        
        Parameters:
        -----------
        tokens : list of str
            List of word tokens
        
        Returns:
        --------
        filtered_tokens : list of str
            Tokens with stopwords removed
        
        Example:
        --------
        >>> processor = TextProcessor()
        >>> tokens = ['this', 'is', 'a', 'great', 'movie']
        >>> processor.remove_stopwords(tokens)
        ['great', 'movie']
        """
        return [token for token in tokens if token.lower() not in self.STOPWORDS]
    
    def process(self, text):
        """
        Full text preprocessing pipeline.
        
        Applies:
        1. clean() : Remove HTML, special chars, normalize
        2. tokenize() : Split into words
        3. remove_stopwords() : Remove common words (if enabled)
        
        Parameters:
        -----------
        text : str
            Raw text to process
        
        Returns:
        --------
        tokens : list of str
            List of processed tokens
        
        Example:
        --------
        >>> processor = TextProcessor(remove_stopwords=True)
        >>> text = "<p>This is a Great movie! 123</p>"
        >>> processor.process(text)
        ['great', 'movie']
        """
        # Clean text
        text = self.clean(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords if enabled (call the token-filtering method)
        if self.remove_stopwords:
            tokens = self.remove_stopwords_tokens(tokens)
        
        # Remove empty tokens
        tokens = [token for token in tokens if token]
        
        return tokens
    
    def __call__(self, text):
        """Allow TextProcessor to be called like a function."""
        return self.process(text)
    
    def __repr__(self):
        """String representation."""
        return f"TextProcessor(remove_stopwords={self.remove_stopwords}, remove_numbers={self.remove_numbers})"
