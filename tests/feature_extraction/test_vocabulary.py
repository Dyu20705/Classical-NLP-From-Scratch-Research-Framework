from src.feature_extraction.vocabulary import Vocabulary


def test_build_and_lookup():
    vocab = Vocabulary()
    vocab.build_from_corpus(["hello", "world", "hello"], max_features=None)
    assert vocab.get_index("hello") != vocab.get_index("world")
    assert vocab.get_word(vocab.get_index("hello")) == "hello"


def test_oov_maps_to_unk():
    vocab = Vocabulary()
    vocab.build_from_corpus(["known"], max_features=None)
    unk_idx = vocab.get_index("<UNK>")
    assert vocab.get_index("unknown_token") == unk_idx


def test_freeze_prevents_add():
    vocab = Vocabulary()
    vocab.add("a")
    vocab.freeze_vocab()
    try:
        vocab.add("b")
    except ValueError:
        return
    assert False, "Expected ValueError when adding to frozen vocab"
