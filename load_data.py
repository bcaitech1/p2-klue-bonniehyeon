import pickle as pickle
import pandas as pd
import torch
import random


# Dataset 구성.
class RE_Dataset(torch.utils.data.Dataset):
  def __init__(self, tokenized_dataset, labels):
    self.tokenized_dataset = tokenized_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

def unkchange(idx_list):
  output,count=torch.unique(idx_list,return_counts=True)
  maxnum=len(idx_list)-count[1]
  picknum = random.randint(0,5)
  random_idx = random.sample(range(1,maxnum-2),picknum)
  for i in random_idx:
    idx_list[i]=3 # 3 = ROBERTA'S UNK TOKEN ID
  return idx_list

# Randomly change token into the UNK token(expecting data cutout effect like image augmentation)
class RE_Dataset_UNK(torch.utils.data.Dataset):
  def __init__(self, tokenized_dataset, labels):
    self.tokenized_dataset = tokenized_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: torch.tensor(unkchange(val[idx])) if key=='input_ids' else torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

# Dataset for dynamic padding
class RE_Dataset_Dynamic(torch.utils.data.Dataset):
  def __init__(self, tokenized_dataset, labels):
    self.tokenized_dataset = tokenized_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx] for key, val in self.tokenized_dataset.items()}
    item['labels'] = self.labels[idx]
    return item

  def __len__(self):
    return len(self.labels)

# for inference
def preprocessing_dataset(dataset, label_type):
  label = []
  for i in dataset[8]:
    if i == 'blind':
      label.append(100)
    else:
      label.append(label_type[i])
  out_dataset = pd.DataFrame({'sentence':dataset[1],'entity_01':dataset[2],'entity_02':dataset[5],'label':label,})
  return out_dataset

def load_data(dataset_dir):
  with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
    label_type = pickle.load(f)
  dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
  dataset = preprocessing_dataset(dataset, label_type)
  
  return dataset

def tokenized_dataset(dataset, tokenizer):
  concat_entity = []
  for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
    temp = ''
    temp = e01 + '@' + e02
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      #truncation='only_second',
      max_length=100,
      add_special_tokens=True,
      )
  return tokenized_sentences

# tokenized dataset for dynamic padding
def tokenized_dataset_Dynamic(dataset, tokenizer):
  concat_entity = []
  for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
    temp = ''
    temp = e01 + '@' + e02
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      add_special_tokens=True,
      )
  return tokenized_sentences
