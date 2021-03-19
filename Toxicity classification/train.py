import os 

import pandas as pd
import numpy as np

import re

from sklearn.model_selection import train_test_split

from tqdm import tqdm
import random 
import argparse

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

from datasets import ToxicityDataset
from model import ToxicityClassifier, ToxicityClassifierEmbedding

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default = './saved_models', help='path for saving checkpoints')
parser.add_argument('--results_dir', type=str, default = './results', help='path for saving results')
parser.add_argument('--data_dir', default = './data', help='data directory')

parser.add_argument('--train',action='store_true')
parser.add_argument('--resume',action='store_true')

parser.add_argument('--learning_rate', type=float, default=0.00002)
parser.add_argument('--adam_epsilon', type=float, default=0.00000001)
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--num_epochs', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--max_length', type=int, default=220)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--start_epoch', type=int, default=1)
parser.add_argument('--num_save_epochs', type=int, default=1)

args = parser.parse_args()


"""Setting seeds for reproducibility"""

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


"""Download the train and test datasets, either from kaggle or from the following links, and place them in args.data_dir folder"""

# train.csv :   "https://drive.google.com/file/d/1--TceffCWOdmOv_oq-ryHn9NDifAwsTo/view?usp=sharing"
# test_public_expanded.csv:  "https://drive.google.com/file/d/1-8yHngJrWfS_cirwbXY7kUYqxkWpRrxB/view?usp=sharing"

# Kaggle link: "https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data"


""" Downsampling the data """

train_raw = pd.read_csv(os.path.join(args.data_dir,'train.csv'))
# print(train_raw.shape)
# print(train_raw.head(10))

test = pd.read_csv(os.path.join(args.data_dir,'test_public_expanded.csv'))
# print(test.shape)
# print(test.head(10))

train_raw['comment_text'] = train_raw['comment_text'].astype(str)
test['comment_text'] = test['comment_text'].astype(str)

train_raw=train_raw.fillna(0)
test = test.fillna(0)

# convert target to 0,1
train_raw['target']=(train_raw['target']>=0.5).astype(float)
test['target']=(test['toxicity']>=0.5).astype(float)

raw_labels = train_raw['target'].values.flatten()
pos_raw = np.where(raw_labels == 1.0)[0]
neg_raw = np.where(raw_labels == 0.0)[0]
white_idx = train_raw[train_raw['white'] >= 0.5].index.values
black_idx = train_raw[train_raw['black'] >= 0.5].index.values
nidx = np.random.choice(neg_raw, len(pos_raw))
pidx = np.random.choice(pos_raw, len(pos_raw))
idxs = np.unique(np.concatenate((nidx, pidx, white_idx, black_idx), axis = 0))

train = train_raw.iloc[idxs]

print(f"{len(train[train['target'] == 1.0 ])} pos; {len(train[train['target'] == 0.0])} neg")


"""Splitting data into train and validation sets"""

X = train['comment_text'].to_numpy()
y = train['target'].to_numpy()
print(X)
print(y)

X_test = test['comment_text'].to_numpy()
print(X_test)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=args.seed)
print(len(X_train),len(y_train))


"""Creating tokenizer and dataloaders"""

torch.backends.cudnn.deterministic = True

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


def create_data_loader(comment_texts,targets=None, tokenizer=None, max_len=128, batch_size=16, shuffle = False, train = True):
  ds = ToxicityDataset(texts=comment_texts, targets=targets, tokenizer=tokenizer, max_len=max_len, train = train)
  return DataLoader(ds, batch_size=batch_size,num_workers=4, shuffle = shuffle)

train_data_loader = create_data_loader(comment_texts = X_train ,targets=y_train, tokenizer=tokenizer, max_len=args.max_length, batch_size=args.batch_size, shuffle=True)
val_data_loader = create_data_loader(comment_texts = X_val ,targets=y_val, tokenizer=tokenizer, max_len=args.max_length, batch_size=args.batch_size, shuffle=True)
test_data_loader = create_data_loader(comment_texts = X_test, tokenizer=tokenizer, max_len=args.max_length, batch_size=args.batch_size, train = False)


"""Defining train and evaulation functions"""

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
  model = model.train()
  losses = []
  correct_predictions = 0
  for d in tqdm(data_loader):
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)
    # print(input_ids, attention_mask)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    _, preds = torch.max(outputs, dim=1)
    # print(input_ids.shape, outputs.shape,preds,targets)

    loss = loss_fn(outputs, targets)
    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    
  return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()
  losses = []
  correct_predictions = 0
  with torch.no_grad():
    for d in tqdm(data_loader):
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)
      outputs = model(input_ids=input_ids,attention_mask=attention_mask)
      _, preds = torch.max(outputs, dim=1)
      # print(input_ids.shape, outputs.shape,preds,targets)
      loss = loss_fn(outputs, targets)
      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())
  return correct_predictions.double() / n_examples, np.mean(losses)


"""Defining and setting models and training params"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ToxicityClassifier(2)
model = model.to(device)


param_optimizer = list(model.named_parameters())
print([n for n, p in param_optimizer])
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
                                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                                ]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
total_steps = len(train_data_loader) * args.num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=total_steps)
loss_fn = nn.CrossEntropyLoss().to(device)

"""Training the model"""

if args.train:
  print("Now training the model")
  print()
  best_accuracy = 0
  history = {'train_acc':[], 'train_loss':[], 'val_acc':[], 'val_loss':[]}


  if not os.path.exists(args.save_dir):
      os.makedirs(args.save_dir)

  for epoch in range(args.num_epochs):
    print(f'Epoch {epoch + 1}/{args.num_epochs}')
    print('-' * 10)
    train_acc, train_loss = train_epoch(model,train_data_loader,loss_fn,optimizer,device,scheduler,len(X_train))
    print(f'Train loss {train_loss} accuracy {train_acc}')
    val_acc, val_loss = eval_model(model,val_data_loader,loss_fn,device,len(X_val))
    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
    torch.save(model.state_dict(), os.path.join(args.save_dir, f'bert_model_epoch_{epoch}.bin'))
    if val_acc > best_accuracy:
      torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_bert_model.bin'))
      best_accuracy = val_acc


"""Getting predictions for test set"""

print("Now Evaluating the model on the test set")


def get_predictions(model, data_loader):
  model = model.eval()
  predictions = []
  with torch.no_grad():
    for d in tqdm(data_loader):
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      outputs = model(input_ids=input_ids,attention_mask=attention_mask)
      _, preds = torch.max(outputs, dim=1)
      predictions.extend(preds)
    
  predictions = torch.stack(predictions).cpu().tolist()
  return np.array(predictions)

state_dict = torch.load(os.path.join(args.save_dir, 'best_bert_model.bin'))
model.load_state_dict(state_dict)
y_predicted=get_predictions(model, test_data_loader)

"""Saving test predictions"""
if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

test['predictions'] = y_predicted.round().astype(int)

test.to_csv(ps.path.join(args.results_dir,'test_predictions.csv'))

'''Saving embeddings for clustering'''

my_model = ToxicityClassifierEmbedding(model)

def get_embeddings(model, data_loader):
  model = model.eval()
  predictions = []
  
  with torch.no_grad():
    for d in tqdm(data_loader):
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      outputs = model(input_ids=input_ids,attention_mask=attention_mask)
      predictions.append(outputs)

  predictions = torch.cat(predictions).cpu().tolist()   

  return np.array(predictions)

X_embed = get_embeddings(my_model, test_data_loader)

with open(os.path.join(args.results_dir, 'second2last.npy'), 'wb') as f:
    np.save(f, X_embed)