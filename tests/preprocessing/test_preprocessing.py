import numpy as np

from src.preprocessing.text_processor import TextProcessor


def test_clean_html_and_spaces():
    p = TextProcessor(remove_stopwords=False)
    text = "<p>Hello   World!</p>"
    assert p.clean(text) == "hello world"


def test_empty_and_none_handling():
    p = TextProcessor()
    assert p.clean("") == ""
    assert p.clean(None) == ""
    assert p.process("") == []


def test_unicode_text_preserved_letters():
    p = TextProcessor()
    tokens = p.process("Tôi yêu NLP và café!")
    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_stopword_removal():
    p = TextProcessor(remove_stopwords=True)
    tokens = p.process("this is a great movie")
    assert tokens == ["great", "movie"]


def test_tokenizer_consistency():
    p = TextProcessor()
    t1 = p.tokenize("a b c")
    t2 = p.process("a b c")
    assert t1 == t2
