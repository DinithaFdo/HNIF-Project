import spacy
import numpy as np
from typing import List, Dict, Any


class LinguisticNoiseFilter:
    """
    Applies NLP techniques to mathematical tensors to zero out structural
    and grammatical noise, isolating true semantic causal triggers.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        print("Initializing Linguistic Noise Filter...")
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"⚠️ spaCy model '{spacy_model}' not found. Downloading...")
            import subprocess

            subprocess.run(["python", "-m", "spacy", "download", spacy_model])
            self.nlp = spacy.load(spacy_model)

        # Define structural tokens used by various Transformer architectures
        self.structural_tokens = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"}

    def process_and_normalize(
        self, raw_tokens: List[str], raw_scores: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Cleans tokens, zeroes out noise, and normalizes scores for the Next.js frontend.
        """
        filtered_data = []
        valid_scores = []

        # Pass 1: Identify noise and clean the text
        for token, score in zip(raw_tokens, raw_scores):
            # Clean DeBERTa/RoBERTa/SentencePiece specific subword prefixes
            # \u2581 is the mathematical unicode for the DeBERTa block character ' '
            clean_token = (
                token.replace("Ġ", "")
                .replace("_", "")
                .replace(" ", "")
                .replace("\u2581", "")
                .strip()
            )

            is_noise = False

            # 1. Check for Transformer Structural Tokens
            if token in self.structural_tokens or clean_token in self.structural_tokens:
                is_noise = True

            # 2. Check for Punctuation or Empty subwords via spaCy
            elif clean_token:
                # We analyze the single clean token
                spacy_doc = self.nlp(clean_token)
                if len(spacy_doc) > 0:
                    spacy_token = spacy_doc[0]
                    if spacy_token.is_punct or spacy_token.is_space:
                        is_noise = True
            else:
                is_noise = True  # Catch empty strings after cleaning

            filtered_data.append(
                {
                    "original_token": token,
                    "display_token": clean_token,
                    "is_noise": is_noise,
                    "raw_score": float(
                        score
                    ),  # Convert numpy float for JSON serialization
                }
            )

            if not is_noise:
                valid_scores.append(float(score))

        # Pass 2: Normalize only the valid semantic scores (Min-Max Scaling)
        # This ensures the highest semantic word gets opacity 1.0 in Next.js
        if valid_scores:
            min_score = min(valid_scores)
            max_score = max(valid_scores)
            range_score = max_score - min_score if max_score > min_score else 1.0

            for item in filtered_data:
                if item["is_noise"]:
                    item["normalized_score"] = 0.0  # Force noise to be invisible in UI
                else:
                    item["normalized_score"] = (
                        item["raw_score"] - min_score
                    ) / range_score
        else:
            # Fallback if everything was flagged as noise
            for item in filtered_data:
                item["normalized_score"] = 0.0

        return filtered_data
