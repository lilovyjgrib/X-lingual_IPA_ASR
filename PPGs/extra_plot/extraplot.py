import re
# similar sounds
merge_arpa= {
    # replace allophones that likely don't exist in Yoruba
    # even if they exist as allophones of other sounds,
    # substituting them would undermine the distributions learned by the model
    # e.g. devoiced shwa could be by essence regarded as [h], but /h/ would never occur in C_C
    'ax-h': 'ax',  # devoiced schwa
    # closures of stops, that occur without release in English codas,
    # however no consonant codas in Yoruba
    'bcl': 'b',
    'dcl': 'd',
    'gcl': 'g',
    'kcl': 'k',
    'pcl': 'p',
    'tcl': 't',

    # replace syllabic with common sonorant labels
    'en': 'n',
    'em': 'm',
    'el': 'l',
    'eng': 'ng',

    ## rhotic vowels
    # 'axr': 'r' ? 'ɹ' ?
    # 'er': 'r', 'ɹ' ?

    # flapped /t, d, n/ substituted for the closest Yoruba analogs
    'dx': 'r',
    'nx': 'n',

    # /h/ sound
    'hh': 'h',
}


diphthongs= ['ey', 'aw', 'ay', 'ow']
diphthong_regex= re.compile('|'.join(sorted(map(re.escape, diphthongs),
                                              key= len, reverse= True)))
oy_regex= re.compile('oy')

def split_diphthongs(label):
  label= ' '.join(label)
  label= diphthong_regex.sub(lambda x: ' '.join(x.group()), label)
  label = oy_regex.sub('oh y', label).split()
  return label



def process_labels_arpa(labels):
  print(labels)
  labels = [[symbol for symbol in label if symbol not in 'h# epi pau'] for label in labels]
  labels = [split_diphthongs(label) for label in labels]
  labels_merge = [[merge_arpa.get(symbol, symbol) for symbol in label] for label in labels]
  labels_final = []

  for label, label_merge in zip(labels, labels_merge):
    label = ' '.join(label)
    label_merge = ' '.join(label_merge)
    for key, item in merge_arpa.items():
      if ' '.join([key, item]) in label:
        label_merge = re.sub(' '.join([item, item]), item, label_merge)

    labels_final.append(label_merge.split())
  print(labels_final[0])
  return labels_final

merge_yor = {'ɛ̃':'ɛ', 'ụ':'u'}

def process_labels_yoruba(labels):

  labels = [item['label'].strip('/') for item in labels]
  labels = [label.split() for label in labels]

  labels = [[phoneme for phoneme in label if phoneme != '|' and phoneme != '||'] for label in labels]
  labels = [[merge_yor[phoneme] if phoneme in merge_yor.keys() else phoneme for phoneme in label] for label in labels]
  yoruba_labels = [' '.join(label) for label in labels]
  return yoruba_labels


def create_ARPAdictionary(labels, include_unk=False):
  '''
  args:
    labels: list of list containing sequence of label for each audio sample
  return: arpa2idx
        dictionary of ARPA_label to index
  '''
  arpas= set()
  for label in labels:
    arpas= arpas.union(set(label))
  arpas= sorted(arpas)

  arpa2idx= {arpa:(idx+2) for idx, arpa in enumerate(arpas)}
  arpa2idx[''] =0 # pad will also be used as this
  if include_unk:
    arpa2idx[''] = len(arpa2idx) + 1

  return arpa2idx, {v: k for k, v in arpa2idx.items()}
