from dataclasses import dataclass
from typing import Iterable

def _base_mappings() -> tuple[dict[str, str], dict[str, str], dict[str, list[str]], dict[str, list[str]]]:
    timit_to_ipa = {
        # vowels
        "aa": "ɑ", "ae": "æ", "ah": "ʌ", "ao": "ɔ",
        "aw": "aw", "ay": "aj", "ax": "ə", "axr": "ə˞",
        "eh": "ɛ", "er": "ɜ˞", "ey": "ej",
        "ih": "ɪ", "ix": "ɨ", "iy": "i",
        "ow": "ow", "oy": "ɔj",
        "uh": "ʊ", "uw": "u", "ux": "ʉ",
        # consonants
        "b": "b", "ch": "t͡ʃ", "d": "d", "dh": "ð", "dx": "ɾ",
        "el": "l̩", "em": "m̩", "en": "n̩",
        "f": "f", "g": "ɡ", "hh": "h", "h": "h",
        "jh": "d͡ʒ", "k": "k", "l": "l", "m": "m", "n": "n",
        "nx": "ɾ̃", "ng": "ŋ", "p": "p", "q": "ʔ", "r": "ɹ",
        "s": "s", "sh": "ʃ", "t": "t", "th": "θ", "v": "v",
        "w": "w", "wh": "ʍ", "y": "j", "z": "z", "zh": "ʒ",
        # prosodic / pauses
        "pau": "|", "epi": "||", "h#": "/",
        # closures (dialectal / marginal)
        "bcl": "b̚", "dcl": "d̚", "gcl": "ɡ̚", "kcl": "k̚", "pcl": "p̚", "tcl": "t̚",
        "ax-h": "ə̥", "eng": "ŋ̍", "hv": "ɦ",
    }
    allophones_substitute = {
        "ax-h": "ə", "bcl": "b", "dcl": "d", "eng": "ŋ", "gcl": "ɡ",
        "hv": "h", "kcl": "k", "pcl": "p", "tcl": "t", "el": "l", "em": "m",
        "en": "n", "hh": "h", "dx": "r", "nx": "n",
    }
    split_diphthongs_IPA = {d: list(d) for d in ["aj", "aw", "ej", "ow", "ɔj"]}
    split_diphthongs_TIMIT = {
        "oy": ["ao", "y"], "ow": ["o", "w"], "ay": ["a", "y"],
        "aw": ["a", "w"], "ey": ["e", "y"],
    }
    return timit_to_ipa, allophones_substitute, split_diphthongs_IPA, split_diphthongs_TIMIT


@dataclass
class EnglishInventoryOptions:
    include_pauses: bool = False
    include_allophones: bool = False
    split_diphthongs: bool = False
    

def timit_inventory(opts: EnglishInventoryOptions) -> list[str]:
    timit_to_ipa, allo_sub, _, split_TIMIT = _base_mappings()
    phones = set(timit_to_ipa.keys())

    if not opts.include_pauses:
        phones -= {"pau", "epi", "h#"}

    if not opts.include_allophones:
        phones -= set(allo_sub)

    if opts.split_diphthongs:
        delete = set(split_TIMIT)
        add = set(x for pair in split_TIMIT.values() for x in pair)
        phones = (phones | add) - delete

    return sorted(phones)


def ipa_inventory(opts: EnglishInventoryOptions) -> list[str]:
    timit_to_ipa, allo_sub, split_IPA, _ = _base_mappings()

    def canon(x: str) -> str:
        return allo_sub.get(x, timit_to_ipa.get(x))

    phones = set(canon(k) if not opts.include_allophones else timit_to_ipa[k]
                 for k in timit_to_ipa.keys())

    if not opts.include_pauses:
        phones -= {"/", "|", "||"}

    if opts.split_diphthongs:
        delete = set(split_IPA.keys())
        add = set(x for pair in split_IPA.values() for x in pair)
        phones = (phones | add) - delete

    return sorted(phones)


def english_to_ipa(
    phones: str | Iterable[str],
    include_allophones: bool = False,
    split_diphthongs: bool = True,
    include_pauses: bool = False,
    sep: str = " "
) -> str:
    timit_to_ipa, allo_sub, split_IPA, split_TIMIT = _base_mappings()

    if isinstance(phones, str):
        toks = [t for t in phones.strip().split() if t]
    else:
        toks = list(phones)

    if split_diphthongs:
        expanded: list[str] = []
        for t in toks:
            if t in split_TIMIT:
                expanded.extend(split_TIMIT[t])
            else:
                expanded.append(t)
        toks = expanded
        timit_to_ipa["a"] = "a"

    out: list[str] = []
    for t in toks:
        ipa = allo_sub.get(t, timit_to_ipa.get(t)) if not include_allophones else timit_to_ipa.get(t)
        if ipa is None:
            raise KeyError(f"Unknown TIMIT phone: {t!r}")
        out.append(ipa)

    res = sep.join(out)
    if not include_pauses:
        res = res.replace("|", "").replace("||", "").replace("/", "").replace("  ", " ").strip()
    return res
