"""
Vocabulary management for natural language processing.

This module provides a Vocabulary class for building and managing word-to-index
mappings commonly used in NLP tasks.
"""

import collections


class Vocabulary:
    """Manage word-to-index and index-to-word mappings."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.freeze = False

    def reverse_vocab(self):
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def add(self, word):
        if self.freeze:
            raise ValueError("Vocabulary is frozen. Cannot add new words.")

        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx

    def get_index(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        if "<UNK>" in self.word2idx:
            return self.word2idx["<UNK>"]
        return 0

    def get_word(self, index):
        if index in self.idx2word:
            return self.idx2word[index]
        if 0 in self.idx2word:
            return self.idx2word[0]
        return "<UNK>"

    def freeze_vocab(self):
        self.freeze = True

    def unfreeze_vocab(self):
        self.freeze = False

    def build_from_corpus(self, words, max_features=None):
        if self.freeze:
            raise ValueError("Vocabulary is frozen. Cannot build from corpus.")

        if "<UNK>" not in self.word2idx:
            self.add("<UNK>")

        word_count = collections.Counter(words)
        most_common = word_count.most_common(max_features)

        for word, _ in most_common:
            if word != "<UNK>":
                self.add(word)

        self.reverse_vocab()

    def size(self):
        return len(self.word2idx)

    def __len__(self):
        return self.size()

    def __repr__(self):
        return f"Vocabulary(size={self.size()}, frozen={self.freeze})"
