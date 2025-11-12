# X-lingual_IPA_ASR
Recognize phonemes in Yoruba through a model trained on phonemes in English

Based on TBD. 

A project for the Universitaet Tuebingen ASR course (2025) 

## What is a phoneme-level model and IPA?

Using a grapheme to phoneme tools or a dataset annotated for phonemes, we train a CTC model to map audio sequence to a phoneme sequence (with no intermediate steps like standard orthography or vocabulary). For interlinguistic compatibility, we convert the phonemes to be represented in [IPA](). 

## English → Yoruba and linguistic generalisation

Regardless of the orthography languages draw their sounds from a universal set of types. Linguists worked out how similar these prototypical sounds are. [link] To a lage extent what sound types a language uses is studied, here we draw data from [PHOIBLE](). This implies that, if two languages use a similar set, some skills in recognizing the sounds of one language can be transferred to the sounds of another. How well? 

---
## `PPGs/`:

This folder contains scripts to extract the embeddings (last layer representation) of correctly predicted phones, as well as code to obtain their dimensionally reduced projection. The script also computes the correlation between the phone representation distance and the distance cost assigned by our fwPER formula.

## `models/`:
This defines the ASRModel class, and provides helper functions for training and evaluation

## `paper/`:
Contains the paper and associated references/ images in Latex format

### 📁 `dataset/`

The subfolder “mfcc_extraction_script” contains the notebooks we used to obtain the log-mels of the audio data. 

Data is available here:

- TIMIT data. Full train set for TIMIT (logmel scale) available at: https://drive.google.com/file/d/13k-ACA6Qt9CJ3MZI6Ot6qD9TAUY3mHUA/view?usp=drive_link
- Original Hugging Face Dataset https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0 
- Yoruba data: https://drive.google.com/file/d/18gejiyyyx3J1jVGnlOyPeFsgoDXVuoZr/view?usp=drive_link

### Workload and tasks

| Aaron Bahr | Nikita Beklemishev | Haejin Cho | Kai Seidenspinner | Ilinca Vandici |
|----|------|------|-------------|--------|
| MFCC extraction| Metrics, G2P adaptation, edit distance algorithms, model training, PPGs | Model design, Model training, Research, Evaluation, Planning   | Overleaf, Planning| MFCC Extraction, G2P Planning, PPGs |



## Authors
Aaron Bahr

Haejin Cho

Ilinca Vandici

Kai Seidenspinner

[Nikita L. Beklemishev](github.com/lilovyjgrib) 

