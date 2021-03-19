import os 

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

import transformers
from transformers import BertTokenizer

class ToxicityDataset(Dataset):
  def __init__(self, texts, targets = None, tokenizer = None, max_len = 128, train = True):
    self.texts = texts
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
    self.train = train

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, item):
    text = self.texts[item]
    encoding = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_len, return_token_type_ids=False,
                                          truncation=True, padding='max_length', return_attention_mask=True, return_tensors='pt',)
    out = {'text': text, 'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten()}

    if self.train:
      out['targets'] = torch.tensor(self.targets[item], dtype=torch.long)
    return out