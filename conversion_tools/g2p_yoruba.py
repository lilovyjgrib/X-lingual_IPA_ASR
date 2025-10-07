"""
AI CODE based on our G2p_yoruba.ipynb 

Minimal Yoruba G2P for a *single* sentence, distilled from the original notebook.
- Requires: epitran
- No CLI, no optional dependency shims (e.g., no dcl).

Pipeline:
1) Epitran transliteration (yor-Latn -> IPA).
2) Normalize tone marks/punctuation and remove orthographic separators.
3) Lowercase, strip final .?!, and wrap with slashes ("/.../").
4) Normalize some nasalized code points to combining-tilde forms.
5) Tokenize into IPA "letters" and separators.
6) Return a space-joined token string.
"""
from __future__ import annotations

import re
from typing import Optional

import epitran  # hard requirement

# Build transliterator once
_EPI = epitran.Epitran("yor-Latn")

# --- Mappings (hardcoded, no dcl) ---
_TONE_VOWS = {
    # acute/grave for a e i o u
    'á': 'a', 'à': 'a',
    'é': 'e', 'è': 'e',
    'í': 'i', 'ì': 'i',
    'ó': 'o', 'ò': 'o',
    'ú': 'u', 'ù': 'u',
    # nasal/tonal n/m
    'ń': 'n', 'ḿ': 'm', 'ǹ': 'n', 'm\u0300': 'm',
    # combining marks for open-mid vowels (ɔ́, ɔ̀, ɛ́, ɛ̀)
    'ɔ\u0301': 'ɔ', 'ɔ\u0300': 'ɔ', 'ɛ\u0301': 'ɛ', 'ɛ\u0300': 'ɛ',
}

_CORRESPONDENCES = {}
_CORRESPONDENCES.update(_TONE_VOWS)
_CORRESPONDENCES.update({
    '. ': '||',
    ',': '|',
    ':': '||',
    '-': '',
    "'": '',
    '‘': '',
    ' ': '',
})

# Normalize nasalized variants
_NORMALIZE_NASALS = {
    'ṹ': 'u\u0303',
    'ũ': 'u\u0303',
    'ĩ': 'i\u0303'
}

_DIALECTAL = {
    'ụ\u0300': 'ʊ', 'ụ\u0301': 'ʊ', 'ụ': 'ʊ',
    'ị\u0300': 'ɪ', 'ị\u0301': 'ɪ', 'ị': 'ɪ'
}

_NOTATION = {
    'ɔ̃': 'ã'
}

# Tokenization pattern: keep multi-char phones and our separators
_TOKEN_REGEX = re.compile(r'k͡p|ɡ͡b|d͡ʒ|ã|ɛ̃|ũ|ĩ|\|\||\w|\||/', flags=re.UNICODE)


def _apply_correspondences(text: str) -> str:
    for dirty, clean in _CORRESPONDENCES.items():
        text = text.replace(dirty, clean)
    return text


def _phoneticize(text: str) -> str:
    return '/' + text.lower().strip('.?!') + '/'


def _normalize_nasals(text: str) -> str:
    for a, b in _NORMALIZE_NASALS.items():
        text = text.replace(a, b)
        
    for a, b in _DIALECTAL.items():
        text = text.replace(a, b)
       
    for a, b in _NOTATION.items():
        text = text.replace(a, b)
    return text


def _to_letters(ipa_string: str) -> list[str]:
    return _TOKEN_REGEX.findall(ipa_string)


def convert(
    text: str,
    *,
    apply_filter: bool = False,
    up_votes: int = 0,
    down_votes: int = 0,
    filter_downvotes_ge: int = 1,
    filter_upvotes_le: int = 2,
) -> Optional[str]:
    """
    Convert ONE Yoruba sentence to the space-joined token string produced by the notebook.

    If `apply_filter=True` and (down_votes >= filter_downvotes_ge and up_votes <= filter_upvotes_le),
    returns None (mimics dataset filtering). Otherwise returns the processed token string.
    """
    if apply_filter and (down_votes >= filter_downvotes_ge and up_votes <= filter_upvotes_le):
        return None

    ipa = _EPI.transliterate(text)
    ipa = _apply_correspondences(ipa)
    ipa = _phoneticize(ipa)
    ipa = _normalize_nasals(ipa)

    tokens = _to_letters(ipa)
    return ' '.join(tokens)

# Convenience alias
process_one = convert
