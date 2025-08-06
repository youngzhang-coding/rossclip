import spacy
from typing import List

class NounExtractor:
    def __init__(self, model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model)
        except OSError:
            print(f"Model '{model}' not found. Downloading...")
            spacy.cli.download(model)
            self.nlp = spacy.load(model)

    def extract_nouns(self, text: str) -> List[str]:
        doc = self.nlp(text)
        tokens = [
            token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"]
        ]
        return list(set(tokens))

def extract_nouns_from_text(text: str, model: str = "en_core_web_sm") -> List[str]:
    extractor = NounExtractor(model)
    return extractor.extract_nouns(text)