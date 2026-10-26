"""Shared email features for prepared course data and raw-message predictions."""

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


PROMOTIONAL_TERMS = ("free", "offer", "winner", "urgent", "discount", "unsubscribe")
FEATURE_NAMES = (
    "character_count", "word_count", "average_word_length", "uppercase_letter_ratio",
    "digit_ratio", "exclamation_count", "currency_symbol_count", "url_count",
    "email_address_count", "subject_length",
) + tuple("term_" + term + "_count" for term in PROMOTIONAL_TERMS)


class EmailFeatures(BaseEstimator, TransformerMixin):
    def fit(self, messages, targets=None):
        return self

    def transform(self, messages):
        if set(FEATURE_NAMES).issubset(messages.columns):
            result = messages.loc[:, list(FEATURE_NAMES)].astype(float).copy()
        else:
            if not {"subject", "body"}.issubset(messages.columns):
                raise ValueError("Provide the 16 prepared features or subject and body columns.")
            rows = []
            for subject, body in messages[["subject", "body"]].fillna("").itertuples(index=False, name=None):
                subject, body = str(subject), str(body)
                content = subject + "\n" + body
                words = re.findall(r"\b\w+\b", content)
                letters = [character for character in content if character.isalpha()]
                rows.append([
                    len(content.strip()), len(words),
                    sum(map(len, words)) / max(1, len(words)),
                    sum(character.isupper() for character in letters) / max(1, len(letters)),
                    sum(character.isdigit() for character in content) / max(1, len(content)),
                    content.count("!"), len(re.findall(r"[$\u20ac\u00a3\u00a5]", content)),
                    len(re.findall(r"\b(?:https?\s*:\s*/\s*/|www\s*\.)", content, re.I)),
                    len(re.findall(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", content)),
                    len(subject),
                    *[len(re.findall(r"\b" + term + r"\b", content, re.I)) for term in PROMOTIONAL_TERMS],
                ])
            result = pd.DataFrame(rows, columns=FEATURE_NAMES, index=messages.index, dtype=float)
        if not np.isfinite(result.to_numpy()).all():
            raise ValueError("Email features must be finite numbers.")
        return result

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_NAMES, dtype=object)