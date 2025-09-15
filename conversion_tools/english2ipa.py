"""
AI CODE based on human-made conversion.py

English (TIMIT) → IPA conversion helpers.
Reuses mappings from conversion.Inventories so no duplication is needed.
"""

from typing import Iterable
from .conversion import Inventories

__EN_CACHE = None


def _get_en_mappings():
    """Lazy-load and cache mappings from the Inventories class."""
    global __EN_CACHE
    if __EN_CACHE is None:
        inv = Inventories()
        (timit_to_ipa,
         allophones_substitute,
         split_diphthongs_IPA,
         split_diphthongs_TIMIT) = inv._init_mappings()
        __EN_CACHE = (timit_to_ipa, allophones_substitute,
                      split_diphthongs_IPA, split_diphthongs_TIMIT)
    return __EN_CACHE


def english_to_ipa(
    phones: str | Iterable[str],
    *,
    substitute_allophones: bool = False,
    split_diphthongs: bool = False,
    on_unknown: str = "keep",   # "keep", "drop", "error"
    sep: str = " ",
) -> str:
    """
    Convert English TIMIT phones to IPA.

    Parameters
    ----------
    phones : str | Iterable[str]
        A whitespace-separated string of TIMIT phones, or an iterable of phones.
    substitute_allophones : bool
        If True, replace marginal/allophonic labels with canonical forms.
    split_diphthongs : bool
        If True, split IPA diphthongs into two symbols (e.g. aɪ → a + j).
    on_unknown : {"keep","drop","error"}
        What to do with unknown phones.
    sep : str
        Separator for output string.

    Returns
    -------
    str
        Converted IPA phones separated by `sep`.
    """
    timit_to_ipa, allo_sub, split_IPA, split_TIMIT = _get_en_mappings()

    if isinstance(phones, str):
        toks = [t for t in phones.strip().split() if t]
    else:
        toks = list(phones)

    out: list[str] = []

    if split_diphthongs:
        expanded: list[str] = []
        for t in toks:
            if t in split_TIMIT:
                expanded.extend(split_TIMIT[t])
            else:
                expanded.append(t)
        toks = expanded


    split_piece_timit_to_ipa = {
        "y": "j",
        "w": "w",
        "oh": "ɔ",
        "a": "a",
        "e": "e",
        "o": "o",
    }

    for t in toks:
        x = allo_sub.get(t, t) if substitute_allophones else t

        ipa = timit_to_ipa.get(x)
        if ipa is None:
            ipa = split_piece_timit_to_ipa.get(x)

        if ipa is None:
            if on_unknown == "keep":
                ipa = x
            elif on_unknown == "drop":
                continue
            elif on_unknown == "error":
                raise KeyError(f"Unknown TIMIT phone: {x!r}")
            else:
                raise ValueError("on_unknown must be one of: keep, drop, error")

        out.append(ipa)

    return sep.join(out)

__all__ = ["english_to_ipa"]
