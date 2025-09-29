from conversion_tools.conversion import *
from model.model import *
from tqdm import tqdm
import re
import torch 
import json
from collections import ChainMap
from itertools import product
from itertools import product 
from scipy import stats
from scipy.spatial.distance import cdist
import seaborn as sns
import matplotlib.pyplot as plt
from conversion_tools.english2ipa import english_to_ipa
from conversion_tools.english2ipa import _get_en_mappings
import os 
import umap 
import gdown
from itertools import combinations
from collections import defaultdict
from pathlib import Path
from itertools import compress
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd 
from conversion_tools.g2p_yoruba import _apply_correspondences, _normalize_nasals

full_path = os.path.abspath(__file__)
path, filename = os.path.split(full_path)
merge_yor = {'ɛ̃':'ɛ', 'ụ':'u'}
def process_labels_ipa(labels):

  labels = [item.strip('/') for item in labels]
  labels = [label.split() for label in labels]

  labels = [[phoneme for phoneme in label if phoneme != '|' and phoneme != '||'] for label in labels]
  labels = [[merge_yor[phoneme] if phoneme in merge_yor.keys() else phoneme for phoneme in label] for label in labels]
  yoruba_labels = [' '.join(label) for label in labels]
  return yoruba_labels

model = ASRModel(
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
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

def load_model():
    checkpoint = torch.load(os.path.join(path, "ResNetCTC.pth.tar"),map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()


load_model()
class Embeddings_Collect():
    def __init__(self, dataset, model):
        self.model = model
        self.dataset_name  = dataset
        #self.loading_fn =  self.load_timit if self.dataset_name == "Timit" else self.load_yoruba
        self.loading_fn = self.load_timit_local if self.dataset_name == "Timit" else self.load_yoruba_local
        self.dataset = self.loading_fn()
        self.inventory = Inventories()
        #self.timit2ipa = self.add_extra_vowels()
        self.labels = [self.dataset[i]['labels'] for i in range(len(self.dataset))] if self.dataset_name == "Timit" else [self.dataset[i]['ipa'] for i in range(len(self.dataset))]
        self.labels_processed = process_labels_ipa([(english_to_ipa(i)) for i in self.labels]) if self.dataset_name == "Timit" else process_labels_ipa(self.labels)
        #self.ipa2timit = {v: k for k, v in self.timit2ipa.items()}
        self.mfccs = [i["mfcc"] for i in self.dataset]
        #self.labels_processed = [[self.timit2ipa[y] for y in x ] for x in process_labels_arpa([i["labels"] for i in self.dataset])] if self.dataset_name =="Timit" else process_labels_yoruba(self.dataset)
        self.mappings_obj = self.load_mappings()
        self.arpa2idx = self.mappings_obj["arpa2idx"]
        self.idx2arpa = self.mappings_obj["idx2arpa"]
        #self.ipadecode =  {idx: self.timit2ipa[arpa] for idx, arpa in self.idx2arpa.items() if arpa in self.timit2ipa}

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


        
    def add_extra_vowels(self):
        self.timit2ipa = self.inventory.timit_to_ipa
        self.timit2ipa['a'] = 'a'
        self.timit2ipa['o'] = 'o'
        self.timit2ipa['e'] = 'e'
        self.timit2ipa['oh'] = 'ɔ'
        return self.timit2ipa



        
    def load_timit(self):
     
        file_id = "1ChXilw3Vo_rXkoGHtkVVaMKxmOoFjGy0"
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
    
    def load_timit_local(self):
        # Load timit locally
        output =  os.path.join(path,"tensor_file_timit.pt")
        timit_data = torch.load(output, weights_only=False)
        for i in timit_data:
            i["labels"] = i["phonemes"]
            del i["phonemes"]
        return timit_data
    
    def load_yoruba_local(self):
        # Load Yoruba locally 
        output = os.path.join(path,"tensor_file_yoruba.pt")
        yoruba_data = torch.load(output, weights_only=False)
        return yoruba_data




    def load_yoruba(self):
        file_id = "1MYBgVY0sbBN30aApR7PyMgNiz0FRVrRA"
        url = f"https://drive.google.com/uc?id={file_id}"
        output = os.path.join(path,"tensor_file_yoruba.pt")
        gdown.download(url, output, quiet=False)
        yoruba_data = torch.load(output, weights_only=False)
        return yoruba_data
    
    

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
        for i, j in tqdm(zip(self.mfccs[0:500], self.labels_processed[0:500]), total=len(self.mfccs[0:500])):
            with torch.no_grad():
                feats = torch.unsqueeze(i,0)
                logits = self.model(feats)
                log_probs = F.log_softmax(logits, dim=-1)
                transcription, embedding = self.decode_ctc(log_probs[0])
                predictions.append(transcription.split(' '))
                embeddings_list.append(embedding)
        self.predictions = predictions 
        
        self.embeddings_list = embeddings_list 
        return predictions
    def embeddings_per_wowel(self):
        """
        self.timit2ipa['a'] = 'a'
        self.timit2ipa['o'] = 'o'
        self.timit2ipa['e'] = 'e'
        self.timit2ipa['oh'] = 'ɔ'
        """
        vowel_embeddings_dict = defaultdict(list,{ k:[] for k in self.vowels_dict.values() })

        for p, l, embedding in  tqdm(zip(self.predictions[0:500], self.labels_processed[0:500], self.embeddings_list[0:500]), total = len(self.embeddings_list[0:500])):
          
            if self.dataset_name == "Timit":
                l = " ".join(l)
            p =english_to_ipa(p)
            
            # Note; Yoruba labels already preprocessed
            # Need to convert predictions to a string tho

            a, score = feature_edit_alignment(p,l)
       
            # Changing this so it takes into account all embeddings
            # Extractable_bools = [True if (a[i][0]==a[i][1]) and (a[i][0] in self.vowels_dict.values()) and (i in list(range(len(p))))  else False for i in range(len(a))]
            # Embeddings of all the matching vowels 
            extractable_bools = [True if (a[i][0]==a[i][1])  and (i in list(range(len(p))))  else False for i in range(len(a))]
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
    
    def get_vowel_df(self):
        vowel_indexes = list()
        X = list()
        
        for i in list(self.vowel_embeddings.keys()):
            for j in self.vowel_embeddings[i]:
                

                X.append(j)
 
                vowel_indexes.append(i)

        X_embedded = np.array(X)
        # Until X_embedded shape is still Dat x 46
        vowel_data = {}
        for i in set(vowel_indexes):
            indexes = [True if vowel_indexes[j] == i else False for j in range(X_embedded.shape[0]) ]
            detached = X_embedded[indexes]
            vowel_data[i] = detached
     
        s = pd.Series(vowel_data).explode()
        # Still proper size
        # Maybe its just bc of the column
            # I just renamed them F1 and F2 to facilitate plotting from Nikita's function
        df = pd.DataFrame(s.to_list(), index=s.index, columns=[str(i) for i in range(X_embedded.shape[1])])
  
        df["Vowel"]= vowel_indexes
        if self.selected_vowels != []:
            df = df.loc[df['column_name'].isin(self.selected_vowels)]
        self.v_df = df
    
        return self.v_df 
    # Get mean for direct plotting 
    def apply_t_SNE(self):
 
        X_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=30).fit_transform(np.array(self.df.values[:, :-1]))
        df_embedded = pd.DataFrame(X_embedded, columns=['F1', 'F2'])
        df_embedded['Vowel'] = self.df.index
        
        self.t_SNE_results = df_embedded
        return df_embedded
    
    def apply_umap(self, df):
        two_dfs = None
        reducer = umap.UMAP()
        vowels = df["Vowel"].reset_index(drop=True)
        #languages = df["Language"].reset_index(drop=True)
        if 'Language' in df.columns:
            two_dfs = True
            languages = df["Language"].reset_index(drop=True)
            df = df.drop(["Language"], axis = 1)

        df = df.drop(["Vowel"], axis = 1)
        #df = df.drop(["Language"], axis = 1)
        data = np.array(df.values)

        X_embedded = reducer.fit_transform(data)
        df_embedded = pd.DataFrame(X_embedded, columns=['F1', 'F2'])
        df_embedded['Vowel'] = vowels
        if two_dfs:
            df_embedded['Language'] = languages

     
        self.umap_results = df_embedded
        df_embedded = self.get_embeddings_mean(df_embedded, two_dfs=two_dfs)
    
        return df_embedded


    def get_canonical_vowels(self):
        return None

    def get_embeddings_mean(self, df, two_dfs):
        if two_dfs:
            return df.groupby('Vowel').agg({'F1': 'mean','F2': 'mean','Language': 'first' }).reset_index()
        else:
            return df.groupby('Vowel').agg("mean").reset_index()
    
    def get_euclidian_distance(self,data, closest = True):
        phonemes = data["Vowel"]
   
        
        np_ver = data.select_dtypes(include='number').to_numpy()
       
      
        df_results = list()
        
        for phone, idx in zip(list(data["Vowel"]), list(range(np_ver.shape[0]))):
            distances = cdist([np_ver[idx]],np_ver, metric='euclidean')[0]
            #Check to see how they are connected
            if closest == True:
                # NOTE: you have to change that so that i gets the closest phoneme w/ no distance
                # Need to mask array before applying argmin
                distances = distances[np.isfinite(distances)]
                print(distances)
                distances = list(data["Vowel"])[np.argmin(distances)]
                df_results.append({"phone":phone,"closest":distances})
            else:
               
               test_values = [i for i in distances]
               distances_annotated ={(phone, other_phone):dist for other_phone, dist in zip(phonemes, distances)}
               df_results.append(distances_annotated)
       
          

        return df_results
    
    def plot(self, data):
        
        #data["Language"] = ["Yoruba"] * len(data)
        palette = sns.color_palette('husl', n_colors=len(set(data["Vowel"].to_list())))

        plt.figure(figsize=(6, 5))

        for i, (vowel, grp) in enumerate(data.groupby('Vowel')):
          
            sns.scatterplot(
            data=grp,
            x='F2', y='F1',
        
            label=vowel,
            s=60,               # marker size
        alpha=0.4
    )
            for _, row in grp.iterrows():
                plt.text(row['F2'] , row['F1'], str(row['Vowel']), fontsize=9, color=palette[i])

      
        plt.legend(title='Vowel', frameon=True, loc='upper right')


        plt.gca()
        plt.gca()
        plt.xlabel('Z1')
        plt.ylabel('Z2')
        plt.title(self.dataset_name+":  "+'Phone Embedding Space')
        plt.tight_layout()
        plt.legend(loc='best')
        plt.show()

def plot_both_languages(data):

    langs = ["English", "Yoruba"]
    palette = sns.color_palette('husl', n_colors=2)
    palette = dict(zip(langs, palette))

    plt.figure(figsize=(6, 5))
   
    for i, (vowel, grp) in enumerate(data.groupby('Vowel')):
           
            
            vowel = vowel.replace('eng_','') if 'eng' in vowel else vowel.replace("yor_","")
            
            sns.scatterplot(
            data=grp,
            x='F2', y='F1',
            color=palette[grp["Language"].values[0]],
            label=vowel,
            s=60,               # marker size
        alpha=0.4
    )
            for _, row in grp.iterrows():
                vowel =  str(row['Vowel']).replace('eng','') if 'eng' in str(row['Vowel']) else  str(row['Vowel']).replace("yor","")
                plt.text(row['F2'] , row['F1'], vowel, fontsize=9, color=palette[grp["Language"].values[0]])

      
    plt.legend(title='Vowel', frameon=True, loc='upper right')


    plt.gca()

    plt.xlabel('Z1')
    plt.ylabel('Z2')
    plt.title('Phone Embedding Space')
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()
    

def get_feature_weights_yor():
    # Feature weights from all Yoruba pairs based on epitran mapping 
    # So it might contains some pairs not existing in our dataset 
    yoruba_ipa_path= os.path.join(path, "yor-Latn.csv")
    yoruba_ipa_map = pd.read_csv(yoruba_ipa_path)
    yoruba_ipa_map = yoruba_ipa_map["Phon"]
    yor = [_apply_correspondences(i) for i in yoruba_ipa_map]
    yor = [_normalize_nasals(i) for i in yor]
    feature_weights = dict()
    for i, j in itertools.combinations(yor, r= 2):
        feat_ = feature_edit_alignment(i,j )
        feat_pair = feat_[0][0]
        score = feat_[1]
        feature_weights[feat_pair] = score
    for i in set(yor):
        feat_ = feature_edit_alignment(i,i )
        feat_pair = feat_[0][0]
        score = feat_[1]
        feature_weights[feat_pair] = score


    return feature_weights

def get_feature_weights_eng():
    timit_to_ipa, allo_sub, split_IPA, split_TIMIT = _get_en_mappings()
    ipa_chars = set(timit_to_ipa.values()) - {'||', '|','/'}
    feature_weights = dict()
    for i, j in itertools.combinations(ipa_chars, r= 2):
        feat_ = feature_edit_alignment(i,j )
        feat_pair = feat_[0][0]
        score = feat_[1]
        feature_weights[feat_pair] = score 
    
    return feature_weights



eng_feature_weights = get_feature_weights_eng()
    

emb = Embeddings_Collect("Yoruba", model=model)

emb.collect_embeddings()

embeddings = emb.embeddings_per_wowel()

plotting = PPGPlot(vowel_embeddings=embeddings, dataset_name="Yoruba", selected_vowels=[])
vowel_df_yor = plotting.get_vowel_df()
umap_yor = plotting.apply_umap(vowel_df_yor)
euclidian =plotting.get_euclidian_distance(umap_yor, closest=False)

print("ok done u can go")

def clean_dict_correlation(yor, preds):
    """This cleans up the keys by ensuring the prediction alignements are the same number as the Yoruba vocabulary
    feature alignments. This excludes repeating alignments (p,f) vs (f,p) but also removes pairs that are not in the 
    predictions by default. This means that non-predicted segments can potentially get excluded.
    
    """
    euclidian_keys = [
    x
    for eucl in preds
    for x in eucl.keys()
]
    yor_feature_keys = list(yor.keys())
    diff_keys = set(euclidian_keys) - set(yor_feature_keys)
    assert len(euclidian_keys) > len(yor_feature_keys)

    flattened_yor = [item for sublist in yor_feature_keys for item in sublist]
    flattened_preds = [item for sublist in preds for item in sublist.keys()]
    flattened_preds = [item for sublist in flattened_preds for item in sublist]
    # Get their combinations
    # Note that is to remove from yoruba and not English
    diff_chars = set(flattened_yor) - set(flattened_preds)

    product_test = itertools.product(diff_chars, flattened_yor)
    # Now remove from yor

    yor_copy = yor_feature_keys.copy()
   
    
    for i in product_test:
        if i in yor_copy:
   
            yor_copy.remove(i)
        if (i[1], i[0]) in yor_copy:
            yor_copy.remove((i[1], i[0]))

  
    # Now remove from euclidian keys
    new_eucl = list()
 
    for e in preds:
        e_copy = e.copy()
        for j in diff_keys:
            
            if j in e.keys():
                
                e_copy.pop(j,None)
            else:
          
                new_eucl.append(e_copy)

    new_eucl = dict(ChainMap(*new_eucl))
    yor_gold = {k:v for k,v  in zip(yor.keys(),yor.values()) if k in yor_copy}
    for i in yor_gold:
        if i [0] ==i[1]:
            yor_gold[i] = np.float64(1)
    for i in new_eucl:
        if i [0] ==i[1]:
            new_eucl[i] = np.float64(1)
   
    eucl_final = {}
    for i in yor_gold:
        eucl_final[i] = new_eucl[i]
    assert len(eucl_final) == len(yor_gold)
    perm = stats.PermutationMethod()
    res = stats.pearsonr(list(eucl_final.values()), list(yor_gold.values()), method = perm)
    return res
    


yor_feature_edit = get_feature_weights_yor()
predicted_eucl, gold_ft = clean_dict_correlation(yor_feature_edit, euclidian)

res = stats.pearsonr(predicted_eucl.values(), gold_ft.values())
print("Results of correlation test")
print(res)

    

emb_timit = Embeddings_Collect("Timit", model= model)
emb_timit.collect_embeddings()
preds=emb_timit.embeddings_per_wowel()


plotting = PPGPlot(vowel_embeddings=preds, dataset_name="Timit", selected_vowels=[])
timit_embeddings =plotting.get_vowel_df()


timit_embeddings.reset_index(drop=True, inplace=True)
vowel_df_yor.reset_index(drop=True, inplace=True)

all_embeddings = pd.concat([timit_embeddings,vowel_df_yor], ignore_index=True)
languages = ["English"] * timit_embeddings.shape[0] + ["Yoruba"] * vowel_df_yor.shape[0]

all_embeddings["Language"] = pd.Series(languages).reset_index(drop=True)
def apply_prefix(df):
    if df["Language"] == "English":
        df["Vowel"] ="eng_"+ df["Vowel"]
        return df
    elif df["Language"] == "Yoruba":
        df["Vowel"] ="yor_"+ df["Vowel"]
        return df

