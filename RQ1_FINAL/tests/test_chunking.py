import numpy as np
from colab_common import Encoder


class MockTok:
    """whitespace 'tokenizer': one token per word, ids are hashes; decode joins back."""
    def __call__(self, text, add_special_tokens=False, truncation=False, verbose=False):
        return {"input_ids": [hash(w) % 10007 for w in text.split()]}
    def __init__(self): self.vocab = {}
    def decode(self, ids, skip_special_tokens=True): return " ".join(f"w{i}" for i in ids)


class MockModel:
    max_seq_length = 20
    tokenizer = MockTok()
    def encode(self, pieces, **kw):
        from sklearn.feature_extraction.text import HashingVectorizer
        return HashingVectorizer(n_features=64, alternate_sign=False, norm="l2").transform(pieces).toarray()


def test_short_text_identical_with_and_without_chunking():
    a, b = Encoder("m", "x", chunk_long=False, model=MockModel()), Encoder("m", "x", chunk_long=True, model=MockModel())
    t = ["short text here"]
    assert np.allclose(a.encode(t, "job"), b.encode(t, "job"))
    assert b.n_chunked == 0


def test_long_text_is_split_pooled_and_normalised():
    long = " ".join(f"word{i}" for i in range(100))
    e = Encoder("m", "x", chunk_long=True, model=MockModel())
    v = e.encode([long], "resume")
    assert e.n_chunked == 1 and e.n_pieces > 1                      # budget = 20-2-4 = 14 tokens per chunk -> 8 chunks
    assert abs(np.linalg.norm(v[0]) - 1.0) < 1e-9
    plain = Encoder("m", "x", chunk_long=False, model=MockModel()).encode([long], "resume")
    assert not np.allclose(v, plain)                                 # sees the whole text, not just the start
    assert "chunking ON" in e.report()
