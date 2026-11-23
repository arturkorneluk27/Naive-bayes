import re

def tokenize(text):

    words = re.split(r"\W+", text)
    return [w.lower() for w in words if len(w) > 2]


def build_vocabulary(documents):
    vocab = set()
    for doc in documents:
        vocab.update(doc)
    return sorted(vocab)
