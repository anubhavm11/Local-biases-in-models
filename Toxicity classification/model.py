import numpy as np

import re
import random 

import torch
from torch import nn

import transformers
from transformers import BertModel, BertTokenizer


class ToxicityClassifier(nn.Module):
  def __init__(self, n_classes):
    super(ToxicityClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)
    self.out = nn.Softmax(dim=-1)

  def forward(self, input_ids, attention_mask):
    # print(input_ids, attention_mask)
    output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    output = self.drop(output[1])
    return self.out(self.linear(output))

class ToxicityClassifierEmbedding(nn.Module):
  def __init__(self, model):
    super(ToxicityClassifierEmbedding, self).__init__()
    self.bert = list(model.children())[0]

  def forward(self, input_ids, attention_mask):
    output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    return output[1]