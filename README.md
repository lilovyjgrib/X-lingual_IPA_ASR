# X-lingual_IPA_ASR
Recognize phonemes in Yoruba through a model trained on phonemes in English

Based on TBD. 

A project for the Universitaet Tuebingen ASR course (2025) 

## What is a phoneme-level model and IPA?

Using a grapheme to phoneme tools or a dataset annotated for phonemes, we train a CTC model to map audio sequence to a phoneme sequence (with no intermediate steps like standard orthography or vocabulary). For interlinguistic compatibility, we convert the phonemes to be represented in [IPA](). 

## English → Yoruba and linguistic generalisation

Regardless of the orthography languages draw their sounds from a universal set of types. Linguists worked out how similar these prototypical sounds are. [link] To a lage extent what sound types a language uses is studied, here we draw data from [PHOIBLE](). This implies that, if two languages use a similar set, some skills in recognizing the sounds of one language can be transferred to the sounds of another. How well? 

---

## 🔧 Current Status

This is a basic project skeleton to get the idea.  
We are currently setting up the pipeline, preprocessing the dataset and planning the logic.

## ✅ To do

- [ ] Preprocess English & Yoruba data, set up target phonemes and correspondences
- [ ] Train phoneme model (CTC) on English
- [ ] Evaluate on Yoruba
- [ ] Analyze the performance, visualize phoneme confusion
- [ ] Report

## Dependencies
\# TBD

## Structure

### `preprocess.ipynb`



### `train.py`

Trains the CTC. TBD

### `predict.py`

Runs a forward pass. TBD

### `utils.py`

- distance calculations ...

### 📁 `results/`

The predictions on the test set are stored in Json here. Visualisations?

### 📁 `dataset/`

- TIMIT data
- https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0 
- processed (TBD)

### \#TBD

## Authors
Aaron 

Haejin Cho

Ilinca Vandici

Kai 

[Nikita L. Beklemishev](github.com/lilovyjgrib) 

