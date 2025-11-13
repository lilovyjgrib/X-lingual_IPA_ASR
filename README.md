# X-lingual_IPA_ASR

This is a final project for the Automatic Speech Recogntion course offered in Summer 2025 by the University of Tuebingen. 
We pretrain a BiLSTM-ResNet model on English language data from TIMIT, and evaluate its performance on Yoruba. We provide measures such as cross-entropy and PMI, hoping to disentangle learning errors from transfer errors. 


## English → Yoruba and linguistic generalisation

Regardless of the orthography languages draw their sounds from a universal set of types. Linguists worked out how similar these prototypical sounds are. [link] To a lage extent what sound types a language uses is studied, here we draw data from [PHOIBLE](). This implies that, if two languages use a similar set, some skills in recognizing the sounds of one language can be transferred to the sounds of another. How well? 

---
### 📁 `PPGs/`:

This folder contains scripts to extract the embeddings (last layer representation) of correctly predicted phones, as well as code to obtain their dimensionally reduced projection. The script also computes the correlation between the phone representation distance and the distance cost assigned by our fwPER formula.

### 📁 `models/`:
This defines the ASRModel class, and provides helper functions for training and evaluation

### 📁 `zero-shot-final/`:
Contains the paper and associated references/ images in Latex format


### 📁 `dataset/`

The subfolder “mfcc_extraction_script” contains the notebooks we used to obtain the log-mels of the audio data. 

Data is available here:

- TIMIT data. Full train set for TIMIT (logmel scale) available at: https://drive.google.com/file/d/13k-ACA6Qt9CJ3MZI6Ot6qD9TAUY3mHUA/view?usp=drive_link
- Original Hugging Face Dataset https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0 
- Yoruba data: https://drive.google.com/file/d/18gejiyyyx3J1jVGnlOyPeFsgoDXVuoZr/view?usp=drive_link

### 📁 `conversion_tools/`
This contains the string processing functions, ensuring the same conventions for English and Yoruba: we convert ARPABET into IPA, and remove inconsistencies in the Yoruba transcription we obtain from G2P. The functions we used for calculating the weighted fwPER measures, based on the initial PanPhon vectors, as well as the edit distance algorithm are also stored in this folder. 

### 📁 `fast_conversion_tools/`
Conversion tools updated: object programming, more features for the feature weights, faster alignment algorithm with AI help. This is an auxiliary file.

### asr+resnet.ipynb

Original model configuration and training settings 

### nikita_results.ipynb

Contains phoneme frequencies, entropy and PMI calculations. 

### G2p_yoruba.ipynb

Original notebook for the conversion of Yoruba Latin script to IPA.

### Workload and tasks

| Aaron Bahr | Nikita Beklemishev | Haejin Cho | Kai Seidenspinner | Ilinca Vandici |
|----|------|------|-------------|--------|
| MFCC extraction, planning, presentation| Everything related to Yoruba, English and phonology, Edit distance algorithms, Evaluation metrics, A bit of everything, Writing | Model design, Model training, Research, Evaluation, Planning, Writing   | Overleaf template, Group meetings participation| MFCC Extraction, G2P Planning, PPGs, Writing |


## Authors
[Aaron Bahr](github.com/AaronB04)

[Haejin Cho](github.com/Mockdd)

[Ilinca Vandici](github.com/ilinkaa)

Kai Seidenspinner

[Nikita L. Beklemishev](github.com/lilovyjgrib) 

