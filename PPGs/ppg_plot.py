from conversion_tools import conversion as cv
from model.model import *
from tqdm import tqdm
import re
import torch 
import json
import seaborn as sns
import matplotlib.pyplot as plt

import os 
from extra_plot.extra_plot import process_labels_arpa, process_labels_yoruba, create_ARPAdictionary
import gdown
from collections import defaultdict
from pathlib import Path
from itertools import compress
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd 
from sklearn.manifold import MDS

full_path = os.path.abspath(__file__)
path, filename = os.path.split(full_path)

class Embeddings_Collect():
    def __init__(self, dataset):
        self.dataset_name  = dataset
        self.loading_fn = self.load_timit if self.dataset_name == "Timit" else self.load_yoruba
        self.dataset = self.loading_fn()
        self.inventory = cv.Inventories()
        self.timit2ipa = self.add_extra_vowels()
        self.ipa2timit = {v: k for k, v in self.timit2ipa.items()}
        self.mfccs = [i["mfcc"] for i in self.dataset]
        self.labels_processed = [[self.timit2ipa[y] for y in x ] for x in process_labels_arpa([i["labels"] for i in self.dataset])] if self.dataset_name =="Timit" else process_labels_yoruba(self.dataset)
        self.mappings_obj = self.load_mappings()
        self.arpa2idx = self.mappings_obj["arpa2idx"]
        self.idx2arpa = self.mappings_obj["idx2arpa"]
        self.ipadecode =  {idx: self.timit2ipa[arpa] for idx, arpa in self.idx2arpa.items() if arpa in self.timit2ipa}

        #Maybe make that part user defined too idk 
        
        self.vowels_dict= {
            # vowels
            "aa": "ɑ",  # father
            "ae": "æ",  # trap
            "ah": "ʌ",  # strut
            "ao": "ɔ",  # thought
            "aw": "aʊ",  # mouth
            "ay": "aɪ",  # price
            "ax": "ə",  # comma (schwa)
            "axr": "ə\u02de",  # nurse (r-colored schwa)
            "eh": "ɛ",  # dress
            "er": "ɜ˞",  # bird (r-colored vowel)
            "ey": "eɪ",  # face
            "ih": "ɪ",  # kit
            "ix": "ɨ",  # high central unrounded (near-schwa)
            "iy": "i",  # fleece
            "ow": "oʊ",  # goat
            "oy": "ɔɪ",  # choice
            "uh": "ʊ",  # foot
            "uw": "u",  # goose
            "ux": "ʉ",  # dude}
            # Added from Haejin code
            "a" :"a",
            "o":"o",
            "oh":"ɔ",
            "e":"e"
}


        self.model = ASRModel(
    ip_channel=39,
    num_classes=46,
    num_res_blocks=3,
    num_cnn_layers=1,
    cnn_filters=50,
    cnn_kernel_size=15, # changing this caused error
    num_rnn_layers=2,
    rnn_dim=128,
    num_dense_layers=1,
    dense_dim=256,
    use_birnn=True,
    rnn_type="lstm",
    rnn_dropout=0.2
)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        
    def add_extra_vowels(self):
        self.timit2ipa = self.inventory.timit_to_ipa
        self.timit2ipa['a'] = 'e'
        self.timit2ipa['o'] = 'o'
        self.timit2ipa['e'] = 'e'
        self.timit2ipa['oh'] = 'ɔ'
        return self.timit2ipa



        
    def load_timit(self):
        file_id = "13k-ACA6Qt9CJ3MZI6Ot6qD9TAUY3mHUA"
        url = f"https://drive.google.com/uc?id={file_id}"
        output =  os.path.join(path,"tensor_file_timit.pt")
        gdown.download(url, output, quiet=False)
        timit_data = torch.load(output, weights_only=False)
        # Rename dictionnary keys
        # Maybe find a faster way to do this ?
        for i in timit_data:
            i["labels"] = i["phonemes"]
            del i["phonemes"]
        
        return timit_data
    def load_yoruba(self):
        file_id = "1AIO2wnXT3DId0fQd7JIWI59TNFFkPREI"
        url = f"https://drive.google.com/uc?id={file_id}"
        output = os.path.join(path,"tensor_file_yoruba.pt")
        gdown.download(url, output, quiet=False)
        yoruba_data = torch.load(output, weights_only=False)
        return yoruba_data
    # Only if TIMIT THO !!
    def load_model(self):
        checkpoint = torch.load(os.path.join(path, "ResNetCTC.pth.tar"),map_location="cpu")
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.model.eval()
    

    def load_mappings(self):
        # This loads arpa2idx, idx2arpa and ipa2idx dictionnaries from a previous pass on TIMIT
        # Since you have to load the entire TIMIT when using createARPADictionnary
        with open(os.path.join(path,'mappings.json')) as f:
            d = json.load(f)
        return dict(d)

    def decode_ctc(self,logits):
        embeddings = []
        pred = torch.argmax(logits, dim=-1).cpu().numpy()
        prev = -1

        output = []
        for idx, p in enumerate(pred):
     
     
            if p != prev and p != 0:  # remove duplicates and blanks
                #YIKES
                output.append(self.idx2arpa[str(p)])
                embeddings.append(logits[idx])
          
            prev = p
        return " ".join(output), embeddings
    
    # Check if possible to do by batch if it"s faster 
    def collect_embeddings(self):
        predictions = []
        embeddings_list = []
        #Add tqdm maybe
        for i, j in tqdm(zip(self.mfccs, self.labels_processed), total=len(self.mfccs)):
            with torch.no_grad():
                feats = torch.unsqueeze(i,0)
                logits = self.model(feats)
                log_probs = F.log_softmax(logits, dim=-1)
                transcription, embedding = self.decode_ctc(log_probs[0])
                predictions.append(transcription.split(' '))
                embeddings_list.append(embedding)
        self.predictions = predictions 
        self.embeddings_list = embeddings_list 

    def embeddings_per_wowel(self):

        self.timit2ipa['a'] = 'e'
        self.timit2ipa['o'] = 'o'
        self.timit2ipa['e'] = 'e'
        self.timit2ipa['oh'] = 'ɔ'
        vowel_embeddings_dict = defaultdict(list,{ k:[] for k in self.vowels_dict.values() })

        for p, l, embedding in  tqdm(zip(self.predictions[0:1000], self.labels_processed[0:1000], self.embeddings_list[0:1000]), total = len(self.embeddings_list[0:1000])):
  # switch stuff over 2 ipa 
            if self.dataset_name == "Timit":
                #l = " ".join([self.timit2ipa[j] for j in l])
                l = " ".join(l)
            p = [self.timit2ipa[i] for i in p ]
            
            # Note; Yoruba labels already preprocessed
            # Need to convert predictions to a string tho
            a, score = cv.feature_edit_alignment(" ".join(p),l)
       
            
            extractable_bools = [True if (a[i][0]==a[i][1]) and (a[i][0] in self.vowels_dict.values()) and (i in list(range(len(p))))  else False for i in range(len(a))]
  # Embeddings of all the matching vowels 

            ids_ = list(compress([i for i in range(len(a))], extractable_bools))
            vowel_embeddings = list(compress(embedding, extractable_bools))
  

            for i,j in zip(ids_, vowel_embeddings):
      
                vowel_embeddings_dict[a[i][0]].append(j)
        self.vowel_embeddings_dict = vowel_embeddings_dict
        return vowel_embeddings_dict




class PPGPlot():
    def __init__(self, dataset_name, vowel_embeddings,selected_vowels):
        self.vowel_embeddings = vowel_embeddings
        self.dataset_name = dataset_name 
        self.selected_vowels = selected_vowels
    

    def plot_canonical_vowels(self): #To get from Nikita script 
        return None 

    def get_vowel_df(self):
        vowel_indexes = list()
        X = list()
        for i in list(self.vowel_embeddings.keys()):
            for j in self.vowel_embeddings[i]:

                X.append(j)
 
                vowel_indexes.append(i)
        X_embedded = np.array(X)
        
        
        vowel_data = {}
        for i in set(vowel_indexes):
            indexes = [True if vowel_indexes[j] == i else False for j in range(X_embedded.shape[0]) ]
            detached = X_embedded[indexes]
            vowel_data[i] = detached
        s = pd.Series(vowel_data).explode()

            # I just renamed them F1 and F2 to facilitate plotting from Nikita's function
        df = pd.DataFrame(s.to_list(), index=s.index, columns=[str(i) for i in range(X_embedded.shape[1])])

        df["Vowel"]= vowel_indexes
        if self.selected_vowels != []:
            df = df.loc[df['column_name'].isin(self.selected_vowels)]
        self.df = df
        return df 
    # Get mean for direct plotting 
    def apply_t_SNE(self):
 
        X_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=30).fit_transform(np.array(self.df.values[:, :-1]))
        df_embedded = pd.DataFrame(X_embedded, columns=['F1', 'F2'])
        df_embedded['Vowel'] = self.df.index
        self.t_SNE_results = df_embedded
        return df_embedded
    
    def apply_PCA(self):
        return None

    """
    def apply_MDS(self):
        MDS_embedding = MDS(n_components=2, n_init=1)
        X_transformed = MDS_embedding.fit_transform(np.array(self.df.values())[:, :-1])
        df_embedded = pd.DataFrame(X_transformed, columns=['F1', 'F2'])
        df_embedded['Vowel'] = self.df['Vowel']
        self.MDS_results = df_embedded
    """
    def get_canonical_vowels(self):
        return None

    def get_embeddings_mean(self, df):
        return df.groupby(['Vowel'])[['F1','F2']].mean().reset_index()
    
    def plot(self, data, canonical):
        data = self.get_embeddings_mean(data)
        
        palette = sns.color_palette('husl', n_colors=len(set(data["Vowel"].to_list())))

        plt.figure(figsize=(6, 5))

        for i, (vowel, grp) in enumerate(data.groupby('Vowel')):
            sns.scatterplot(
            data=grp,
            x='F2', y='F1',
            color=palette[i],
            label=vowel,
            s=60,               # marker size
        alpha=0.6
    )
            for _, row in grp.iterrows():
                plt.text(row['F2'] + 10, row['F1'], str(row['Vowel']), fontsize=9, color=palette[i])

        """
        if canonical :
            sns.kdeplot(x=grp.F2, y=grp.F1,
                levels=[0.5],
                color=palette[i],
                linewidths=2,
                alpha=0.5,
                label=vowel)
        """
        plt.legend(title='Vowel', frameon=True, loc='upper right')

# Phonetic convention: F1 ↑ downwards, F2 ↑ leftwards
        #plt.gca().invert_yaxis()
        #plt.gca().invert_xaxis()
        # Assuming no axis inversion
        plt.gca()
        plt.gca()
        plt.xlabel('X1')
        plt.ylabel('X1')
        plt.title(self.dataset_name+":  "+'Vowel Space')
        plt.tight_layout()
        plt.show()


emb = Embeddings_Collect("Timit")
emb.load_model()

emb.collect_embeddings()

embeddings = emb.embeddings_per_wowel()

plotting = PPGPlot(vowel_embeddings=embeddings, dataset_name="Timit", selected_vowels=[])
plotting.get_vowel_df()
# Can use this returned df for similairty analysis 
t_sne_res = plotting.apply_t_SNE()
plotting.plot(t_sne_res, canonical=False)
