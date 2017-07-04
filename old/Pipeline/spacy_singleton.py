import spacy

_nlp_model = None


def get_nlp_model():
    global _nlp_model
    if not _nlp_model:
        _nlp_model = spacy.load('en')
    return _nlp_model