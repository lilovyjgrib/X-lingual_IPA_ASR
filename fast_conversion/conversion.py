import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Sequence, Iterable, Optional, Literal

from panphon import FeatureTable

from .english2ipa import EnglishInventoryOptions, ipa_inventory
from .g2p_yoruba import YorubaInventoryOptions, yoruba_inventory

Symbol = str
Vec = np.ndarray  # feature vector of shape (F) in {-1,0,1}


@dataclass
class CostModel:
    ndim: int
    sub_weights: np.ndarray = field(init=False)
    del_mult: np.ndarray = field(init=False)
    ins_mult: np.ndarray = field(init=False)
    del_base: float = 1.0
    ins_base: float = 1.0
    normalize: bool | None = None

    def __post_init__(self):
        mk = lambda: np.ones(self.ndim, np.float32)
        self.sub_weights = mk()
        self.del_mult = mk()
        self.ins_mult = mk()

@dataclass
class CostTables:
    pred_alphabet: list[Symbol]
    gold_alphabet: list[Symbol]
    pred2idx: dict[Symbol, int]
    gold2idx: dict[Symbol, int]
    sub: np.ndarray 
    dcost: np.ndarray 
    icost: np.ndarray 

    def show(self, empty: str = "∅") -> pd.DataFrame:
        idx = self.pred_alphabet + [empty]
        cols = self.gold_alphabet + [empty]
        M = np.zeros((len(idx), len(cols)), dtype=np.float32)
        M[:len(self.pred_alphabet), :len(self.gold_alphabet)] = self.sub
        M[:len(self.pred_alphabet), -1] = self.dcost
        M[-1, :len(self.gold_alphabet)] = self.icost
        df = pd.DataFrame(M, index=idx, columns=cols)
        df.index.name = f"Predicted"
        df.columns.name = f"Gold"
        return df


def _build_cost_tables(pred_alphabet, gold_alphabet, pred_sym2vec, gold_sym2vec, cm) -> CostTables:
    pred2idx = {s: i for i, s in enumerate(pred_alphabet)}
    gold2idx = {s: i for i, s in enumerate(gold_alphabet)}

    Vp = np.stack([pred_sym2vec[s] for s in pred_alphabet], axis=0)
    Vg = np.stack([gold_sym2vec[s] for s in gold_alphabet], axis=0)
    
    def _del_cost(v: np.ndarray) -> float:
        active = cm.del_mult[(v == 1)]
        if active.size == 0: return float(cm.del_base)
        return float(cm.del_base * np.min(active))
    
    def _ins_cost(v: np.ndarray) -> float:
        active = cm.ins_mult[(v == 1)]
        if active.size == 0: return float(cm.ins_base)
        return float(cm.ins_base * np.min(active))

    dcost = np.array([_del_cost(v) for v in Vp], dtype=np.float32)
    icost = np.array([_ins_cost(v) for v in Vg], dtype=np.float32)

    Vp_exp = Vp[:, None, :]
    Vg_exp = Vg[None, :, :]
    union = (Vp_exp != 0) | (Vg_exp != 0)
    mism  = (Vp_exp != Vg_exp) & union
    W = cm.sub_weights[None, None, :]
    num = (mism  * W).sum(axis=2)
    den = (union * W).sum(axis=2)
    sub = (num / den).astype(np.float32)

    if cm.normalize:
        sub = sub / float(np.max(sub))

    return CostTables(
        pred_alphabet=list(pred_alphabet),
        gold_alphabet=list(gold_alphabet),
        pred2idx=pred2idx,
        gold2idx=gold2idx,
        sub=sub,
        dcost=dcost,
        icost=icost,
    )


@dataclass
class AlignResult:
    distance: float
    alignment: list[tuple[Symbol, Symbol]]
    src: list[Symbol]
    tgt: list[Symbol]
    empty: str = "∅"

    def show(self, src_name: str = "ENG", tgt_name: str = "YOR") -> None:
        total = len([t for _, t in self.alignment if t != self.empty])
        print(f"\nDistance: {round(self.distance, 4)}")
        print(f" {src_name} -> {' '.join(self.src)}")
        print(f" {tgt_name} -> {' '.join(self.tgt)}")
        for a, b in self.alignment:
            print(f"{a:5} → {b:5}")
        if total > 0:
            print(f"PER: {round(self.distance / total, 4)}")


class FastAligner:

    def __init__(self):
        self.pred_alphabet: list[Symbol] = []
        self.gold_alphabet: list[Symbol] = []
        self.pred_feats: dict[Symbol, Vec] = {}
        self.gold_feats: dict[Symbol, Vec] = {}
        self.ndim: int = 19
        self.cm: CostModel = CostModel(ndim=self.ndim)
        self.tables: Optional[CostTables] = None

    def set_alphabets(self, pred_alphabet: Sequence[Symbol] | str, gold_alphabet: Sequence[Symbol] | str):
        if isinstance(pred_alphabet, str): pred_alphabet = pred_alphabet.split()
        if isinstance(gold_alphabet, str): gold_alphabet = gold_alphabet.split()
        self.pred_alphabet = list(pred_alphabet)
        self.gold_alphabet = list(gold_alphabet)
        return self
    
    def alphabets_from_project(self, english_opts: Optional[EnglishInventoryOptions] = None,
                               yoruba_opts: Optional[YorubaInventoryOptions] = None):
        self.pred_alphabet = list(ipa_inventory(english_opts))
        self.gold_alphabet = list(yoruba_inventory(yoruba_opts))
        return self

    def set_feature_maps(self, sym2vec: dict[str, Vec]):
        assert self.pred_alphabet and self.gold_alphabet, "alphabets not set"
        first_key = next(iter(sym2vec))
        self.ndim = int(np.asarray(sym2vec[first_key]).reshape(-1).shape[0])
        
        missing_pred = [s for s in self.pred_alphabet if s not in sym2vec]
        missing_gold = [s for s in self.gold_alphabet if s not in sym2vec]
        assert not missing_pred and not missing_gold, f"missing symbols: pred={missing_pred}, gold={missing_gold}"
        
        def _get_vec(sym: str) -> np.ndarray:
            v = np.asarray(sym2vec[sym], dtype=np.int8).reshape(-1)
            assert v.shape[0] == self.ndim, f"vector length mismatch for {sym!r}: {v.shape[0]} != {self.ndim}"
            return v

        self.pred_feats = {s: _get_vec(s) for s in self.pred_alphabet}
        self.gold_feats = {s: _get_vec(s) for s in self.gold_alphabet}
        return self
        
    def feature_maps_from_panphon(self, drop_feats: Optional[Sequence[str]] = None) -> "FastAligner":
        assert self.pred_alphabet and self.gold_alphabet, "alphabets not set"
        phones = list(set(self.pred_alphabet + self.gold_alphabet))
        
        ft = FeatureTable()
        feats = list(dict(ft.word_fts(phones[0])[0]))
        drop_feats = list(drop_feats or [])
        drop_idx = [feats.index(f) for f in drop_feats]
        self.ndim = len(feats) - len(drop_feats)

        def _get_vec(p: str) -> np.ndarray:
            v = ft.word_fts(p)
            assert len(v) == 1, f"'{p}' is not a single phone"
            v = np.asarray(v[0].numeric(), dtype=np.int8)
            if drop_idx:
                v = np.delete(v, drop_idx)
            assert v.shape[0] == self.ndim, f"vector length mismatch for {sym!r}: {v.shape[0]} != {self.ndim}"
            return v

        self.pred_feats = {s: _get_vec(s) for s in self.pred_alphabet}
        self.gold_feats = {s: _get_vec(s) for s in self.gold_alphabet}
        
        return self

    def set_cost_model(self, cm: CostModel):
        self.cm = cm
        self.cm.ndim = self.ndim
        return self

    def build_tables(self):
        assert self.pred_alphabet and self.gold_alphabet, "alphabets not set"
        assert self.pred_feats and self.gold_feats, "feature maps not set"
        self.tables = _build_cost_tables(self.pred_alphabet, self.gold_alphabet, self.pred_feats, self.gold_feats, self.cm)
        return self


    def align(self, A: str | Sequence[Symbol], B: str | Sequence[Symbol], empty: str = "∅") -> AlignResult:
        assert self.tables is not None, "tables not built"
        A_syms = A.split() if isinstance(A, str) else list(A)
        B_syms = B.split() if isinstance(B, str) else list(B)
        A_idx = np.array([self.tables.pred2idx[s] for s in A_syms], dtype=np.int32)
        B_idx = np.array([self.tables.gold2idx[s] for s in B_syms], dtype=np.int32)
        n, m = len(A_idx), len(B_idx)
        D = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
        P = np.zeros((n + 1, m + 1), dtype=np.uint8)  # 0=diag,1=up,2=left
        D[0, 0] = 0.0
        for i in range(1, n + 1):
            D[i, 0] = D[i - 1, 0] + self.tables.dcost[A_idx[i - 1]]; P[i, 0] = 1
        for j in range(1, m + 1):
            D[0, j] = D[0, j - 1] + self.tables.icost[B_idx[j - 1]]; P[0, j] = 2
        for i in range(1, n + 1):
            ai = A_idx[i - 1]
            for j in range(1, m + 1):
                bj = B_idx[j - 1]
                d_diag = D[i - 1, j - 1] + self.tables.sub[ai, bj]
                d_up   = D[i - 1, j]     + self.tables.dcost[ai]
                d_left = D[i, j - 1]     + self.tables.icost[bj]
                if d_diag <= d_up and d_diag <= d_left:
                    D[i, j] = d_diag; P[i, j] = 0
                elif d_up <= d_left:
                    D[i, j] = d_up;   P[i, j] = 1
                else:
                    D[i, j] = d_left; P[i, j] = 2
        i, j = n, m
        align: list[tuple[Symbol, Symbol]] = []
        while i > 0 or j > 0:
            move = P[i, j]
            if i > 0 and j > 0 and move == 0:
                align.append((A_syms[i - 1], B_syms[j - 1])); i -= 1; j -= 1
            elif i > 0 and (j == 0 or move == 1):
                align.append((A_syms[i - 1], empty)); i -= 1
            else:
                align.append((empty, B_syms[j - 1])); j -= 1
        align.reverse()
        return AlignResult(distance=float(D[n, m]), alignment=align, src=A_syms, tgt=B_syms, empty=empty)

    def per(self, predicted_sents: Sequence[str], golden_sents: Sequence[str], empty: str = "∅") -> float:
        assert len(predicted_sents) == len(golden_sents), "two lists don't align"
        edits = 0.0; denom = 0
        for p, g in zip(predicted_sents, golden_sents):
            res = self.align(p, g, empty=empty)
            edits += res.distance
            denom += len(res.tgt)
        return edits / denom

    def confusion_matrix(self, predicted_sents: Sequence[str], golden_sents: Sequence[str],
                         form: Literal["counts","joint","given_gold","given_pred"] = "counts", empty: str = "∅") -> pd.DataFrame:
        assert len(predicted_sents) == len(golden_sents), "two lists don't align"
        alignments: list[tuple[str, str]] = []
        for p, g in zip(predicted_sents, golden_sents):
            res = self.align(p, g, empty=empty)
            alignments.extend(res.alignment)
        cooc = pd.DataFrame(0, index=self.pred_alphabet+[empty], columns=self.gold_alphabet+[empty])
        pi = {lab: i for i, lab in enumerate(cooc.index)}
        gi = {lab: j for j, lab in enumerate(cooc.columns)}
        M = cooc.values
        for src, tgt in alignments:
            M[pi[src], gi[tgt]] += 1
        match form:
            case "counts": return cooc
            case "joint": return cooc.div(cooc.values.sum())
            case "given_gold": return cooc.div(cooc.sum(axis=0).replace(0, np.nan), axis=1).fillna(0)
            case "given_pred": return cooc.div(cooc.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    def align_dataset(self, predicted_sents: Sequence[str], golden_sents: Sequence[str], empty: str = "∅") -> pd.DataFrame:
        rows: list[dict] = []
        for p, g in zip(predicted_sents, golden_sents):
            res = self.align(p, g, empty=empty)
            rows.append({"Model_pred": p, "Yoruba_label": g, "Alignment": res.alignment,
                         "Distance": res.distance, "Score": res.distance / len(res.tgt), "All": res})
        return pd.DataFrame(rows)
