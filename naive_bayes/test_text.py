import unittest

from text import tokenize, build_vocabulary


class TestTokenize(unittest.TestCase):

    def test_lowercase(self):
        self.assertEqual(tokenize("Hello WORLD"), ["hello", "world"])
    
    def test_punctuation(self):
        self.assertEqual(tokenize("Hello, world!!!"), ["hello", "world"])

    def test_short_words_removed(self):
        self.assertEqual(tokenize("I go to the park"), ["the", "park"])

    def test_empty_text(self):
        self.assertEqual(tokenize(""), [])

    def test_example(self):
        docs = [tokenize("Click HERE to win!"), tokenize("Win a free dog, click now")]
        self.assertEqual(docs, [["click", "here", "win"],
                                ["win", "free", "dog", "click", "now"]])


class TestBuildVocabulary(unittest.TestCase):

    def test_no_duplicates(self):
        docs = [["dog", "cat"], ["dog", "bird"]]
        self.assertEqual(build_vocabulary(docs), ["bird", "cat", "dog"])

    def test_sorted(self):
        vocab = build_vocabulary([["zebra", "apple", "mango"]])
        self.assertEqual(vocab, sorted(vocab))

    def test_empty(self):
        self.assertEqual(build_vocabulary([]), [])

if __name__ == "__main__":
    unittest.main()