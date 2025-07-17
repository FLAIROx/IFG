"""Calculates self-BLEU scores to measure diversity between generated texts."""

import numpy as np
import string
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from transformers import AutoTokenizer

# Initialize tokenizer globally to avoid reloading
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)


def is_valid_token(token):
    """Check if token is not empty and not just punctuation"""
    if not token.strip():
        return False
    if all(char in string.punctuation for char in token):
        return False
    return True


def get_sentence_bleu(references, sentence):
    """
    Return sentence bleu w.r.t to reference sentences.
    Args:
        references (list(list(str))): List of tokenized references
        sentence (list(str)): Tokenized sentence to compare to the references
    """
    try:
        score = sentence_bleu(
            references, sentence, smoothing_function=SmoothingFunction().method1
        )
    except (KeyError, ZeroDivisionError):
        score = 0

    return score


def get_self_bleu(all_solutions):
    """Return self-BLEU score (0 = super diverse, 1=not diverse)"""

    # Tokenize each solution into token IDs
    all_solutions_tokens = [
        tokenizer.encode(s.lower(), add_special_tokens=False) for s in all_solutions
    ]

    # Convert token IDs to token strings and filter out invalid tokens
    all_solutions = [
        [
            tokenizer.decode(token, skip_special_tokens=True)
            for token in solution
            if is_valid_token(tokenizer.decode(token, skip_special_tokens=True))
        ]
        for solution in all_solutions_tokens
    ]

    bleu_scores = []
    for i, sol in enumerate(all_solutions):
        references = all_solutions[:i] + all_solutions[i + 1 :]
        score = get_sentence_bleu(references, sol)
        bleu_scores.append(1 - score)

    return np.mean(bleu_scores) if bleu_scores else 0
