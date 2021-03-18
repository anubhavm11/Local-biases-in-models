
import sys, os, csv, codecs, numpy as np, pandas as pd
import json, string, random, time, pickle, gc, pdb
from PIL import Image
from PIL import ImageFilter
import numpy as np
import random
from random import shuffle
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

import argparse

from tqdm import tqdm

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms

import torch.nn.functional as F
import torchvision.models as models
import torch.nn.utils

from datasets import CocoObjectGender
from model import ObjectMultiLabel, ObjectMultiLabelEncoder
from logger import Logger

"""# Setting Parameters"""

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default = './results', help='path for saving checkpoints')
parser.add_argument('--log_dir', type=str, default = './results', help='path for saving log files')

parser.add_argument('--ratio', type=str, default = '0')
parser.add_argument('--num_object', type=int, default = 79)

parser.add_argument('--annotation_dir', type=str, default='./data', help='annotation files path')
parser.add_argument('--image_dir', default = './data', help='image directory')

parser.add_argument('--balanced', action='store_true', help='use balanced subset for training, ratio will be 1/2/3')
parser.add_argument('--gender_balanced', action='store_true', help='use gender balanced subset for training')
parser.add_argument('--batch_balanced', action='store_true', help='in every batch, gender balanced')

parser.add_argument('--blackout', action='store_true')
parser.add_argument('--blackout_box', action='store_true')
parser.add_argument('--blackout_face', action='store_true')
parser.add_argument('--blur', action='store_true')
parser.add_argument('--grayscale', action='store_true')
parser.add_argument('--edges', action='store_true')

parser.add_argument('--no_image', action='store_true', help='do not load image in dataloaders')

parser.add_argument('--train',action='store_true', default=True)
parser.add_argument('--resume',action='store_true')
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--start_epoch', type=int, default=1)
parser.add_argument('--num_save_epochs', type=int, default=10)

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

"""Download the train and validation datasets from the following links, unzip them and place them in args.image_dir"""

# Train Images:	 	"http://images.cocodataset.org/zips/train2014.zip"
# Validation Images:	"http://images.cocodataset.org/zips/val2014.zip"
# Annotation files are present in ./data and taken from https://github.com/uvavision/Balanced-Datasets-Are-Not-Enough.git


"""Defining dataset class and creating dataloaders"""

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                 std = [0.229, 0.224, 0.225])
# Image preprocessing
train_transform = transforms.Compose([
    transforms.Resize(args.image_size),
    transforms.RandomCrop(args.crop_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize])
val_transform = transforms.Compose([
    transforms.Resize(args.image_size),
    transforms.CenterCrop(args.crop_size),
    transforms.ToTensor(),
    normalize])


"""Define Data Loaders"""

train_data = CocoObjectGender(args, annotation_dir = args.annotation_dir, \
        image_dir = args.image_dir, split = 'train', transform = train_transform)
val_data = CocoObjectGender(args, annotation_dir = args.annotation_dir, \
        image_dir = args.image_dir, split = 'val', transform = val_transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size,
            shuffle = True, num_workers = 2, pin_memory = True)

val_loader = torch.utils.data.DataLoader(val_data, batch_size = args.batch_size,
            shuffle = False, num_workers = 2, pin_memory = True)


"""Define model parameters and initialize model"""

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = ObjectMultiLabel(args, args.num_object).to(device)

object_weights = torch.FloatTensor(train_data.getObjectWeights())
criterion = nn.BCEWithLogitsLoss(weight=object_weights, reduction='mean').to(device)

# print model
def trainable_params():
    for param in model.parameters():
        if param.requires_grad:
            yield param

num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('num_trainable_params:', num_trainable_params)
optimizer = torch.optim.Adam(trainable_params(), args.learning_rate, weight_decay = 1e-5)


"""Training and test functions"""

def train(args, epoch, model, criterion, train_loader, optimizer, train_logger, device, logging=True):
    model.train()
    nTrain = len(train_loader.dataset) # number of images
    loss_logger = AverageMeter()

    res = list()
    t = tqdm(train_loader, desc = 'Train %d' % epoch)
    for batch_idx, (images, targets, genders, image_ids) in enumerate(t):

        # Set mini-batch dataset
        if args.batch_balanced:
            man_idx = genders[:, 0].nonzero().squeeze()
            if len(man_idx.size()) == 0: man_idx = man_idx.unsqueeze(0)
            woman_idx = genders[:, 1].nonzero().squeeze()
            if len(woman_idx.size()) == 0: woman_idx = woman_idx.unsqueeze(0)
            selected_num = min(len(man_idx), len(woman_idx))

            if selected_num < args.batch_size / 2:
                continue
            else:
                selected_num = args.batch_size / 2
                selected_idx = torch.cat((man_idx[:selected_num], woman_idx[:selected_num]), 0)

            images = torch.index_select(images, 0, selected_idx)
            targets = torch.index_select(targets, 0, selected_idx)
            genders = torch.index_select(genders, 0, selected_idx)

        images = images.to(device)
        targets = targets.to(device)

        # Forward, Backward and Optimizer
        preds = model(images)

        loss = criterion(preds, targets)
        loss_logger.update(loss.item())

        preds = torch.sigmoid(preds)
        res.append((image_ids, preds.data.cpu(), targets.data.cpu(), genders))

        # backpropogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print log info
        t.set_postfix(loss = loss_logger.avg)

    total_preds   = torch.cat([entry[1] for entry in res], 0)
    total_preds_score, total_preds_val = torch.max(total_preds, dim = 1)

    total_targets = torch.cat([entry[2] for entry in res], 0)
    total_targets_score, total_targets_val = torch.max(total_targets, dim = 1)

    total_genders = torch.cat([entry[3] for entry in res], 0)

    task_f1_score = f1_score(total_targets.numpy(), (total_preds >= 0.5).long().numpy(), average = 'macro')
    meanAP = average_precision_score(total_targets.numpy(), total_preds.numpy(), average='macro')
    
    if logging:
        train_logger.scalar_summary('loss', loss_logger.avg, epoch)
        train_logger.scalar_summary('task_f1_score', task_f1_score, epoch)
        train_logger.scalar_summary('meanAP', meanAP, epoch)

def test(args, epoch, model, criterion, val_loader, val_logger, device, logging=True):

    model.eval()
    nVal = len(val_loader.dataset) # number of images
    loss_logger = AverageMeter()

    res = list()
    t = tqdm(val_loader, desc = 'Val %d' % epoch)

    for batch_idx, (images, targets, genders, image_ids) in enumerate(t):
        # Set mini-batch dataset
        images = images.to(device)
        targets = targets.to(device)

        preds = model(images)

        loss = criterion(preds, targets)
        loss_logger.update(loss.item())

        preds = torch.sigmoid(preds)

        res.append((image_ids, preds.data.cpu(), targets.data.cpu(), genders))

        # Print log info
        t.set_postfix(loss = loss_logger.avg)

    total_preds   = torch.cat([entry[1] for entry in res], 0)
    total_preds_score, total_preds_val = torch.max(total_preds, dim = 1)

    total_targets = torch.cat([entry[2] for entry in res], 0)
    total_targets_score, total_targets_val = torch.max(total_targets, dim = 1)

    total_genders = torch.cat([entry[3] for entry in res], 0)

    task_f1_score = f1_score(total_targets.numpy(), (total_preds >= 0.5).long().numpy(), average = 'macro')
    meanAP = average_precision_score(total_targets.numpy(), total_preds.numpy(), average='macro')

    if logging:
        val_logger.scalar_summary('loss', loss_logger.avg, epoch)
        val_logger.scalar_summary('task_f1_score', task_f1_score, epoch)
        val_logger.scalar_summary('meanAP', meanAP, epoch)

    return task_f1_score

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


"""Setup everything else"""

if not os.path.exists(args.save_dir):
  os.makedirs(args.save_dir)

if not os.path.exists(args.log_dir):
  os.makedirs(args.log_dir)

train_log_dir = os.path.join(args.log_dir, 'train')
val_log_dir = os.path.join(args.log_dir, 'val')

train_logger = Logger(train_log_dir)
val_logger = Logger(val_log_dir)


"""Checking for resume"""

best_performance = 0
if args.resume:
    if os.path.isfile(os.path.join(args.save_dir, 'checkpoint.pth.tar')):
        print("=> loading checkpoint '{}'".format(args.save_dir))
        checkpoint = torch.load(os.path.join(args.save_dir, 'checkpoint.pth.tar'))
        args.start_epoch = checkpoint['epoch']
        best_performance = checkpoint['best_performance']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.save_dir))

print('before training, evaluate the model')
test(args, 0, model, criterion, val_loader, None, device, logging=False)


"""Training the model"""
if args.train:
	for epoch in range(args.start_epoch, args.num_epochs + 1):
	  print(f"Training Epoch = {epoch}")
	  train(args, epoch, model, criterion, train_loader, optimizer, train_logger, device, logging=True)
	  cur_score = test(args, epoch, model, criterion, val_loader, val_logger, device, logging=True)
	  is_best = cur_score > best_performance
	  print(f"validation score = {cur_score}"+(" (Best so far)" if is_best else ""))
	  best_performance = max(cur_score, best_performance)
	  model_state = {
	      'epoch': epoch + 1,
	      'state_dict': model.state_dict(),
	      'best_performance': best_performance}

	  if (epoch%args.num_save_epochs == 0):
	    torch.save(model_state, os.path.join(args.save_dir, f'checkpoint_{epoch}.pth.tar'))
	  
	  if (is_best):
	    torch.save(model_state, os.path.join(args.save_dir, f'best_model.pth.tar'))


""" Load the best model """

model2 = ObjectMultiLabelEncoder(args, args.num_object).to(device)

if os.path.isfile(os.path.join(args.save_dir, 'best_model.pth.tar')):
  print("=> loading checkpoint '{}'".format(args.save_dir))
  checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pth.tar'))
  args.start_epoch = checkpoint['epoch']
  best_performance = checkpoint['best_performance']
  model2.load_state_dict(checkpoint['state_dict'])
  print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
else:
  print("=> no checkpoint found at '{}'".format(args.save_dir))



""" Save test predictions and embeddings """

def get_embeddings(args, model, val_loader, device):
    model.eval()
    nVal = len(val_loader.dataset) # number of images
    epoch = 0
    res = list()
    t = tqdm(val_loader, desc = 'Val %d' % epoch)

    for batch_idx, (images, targets, genders, image_ids) in enumerate(t):
        # if batch_idx == 100: break # for debugging

        # Set mini-batch dataset
        images = images.to(device)
        targets = targets.to(device)

        img_features, preds = model(images)

        preds = torch.sigmoid(preds)

        res.append((image_ids, preds.data.cpu(), targets.data.cpu(), genders, img_features.cpu()))

    total_img_ids   = torch.cat([entry[0] for entry in res], 0).numpy()

    total_preds_scores   = torch.cat([entry[1] for entry in res], 0)
    total_preds = (total_preds_scores>=0.5).long().numpy()
    total_preds_scores = total_preds_scores.numpy()

    total_targets = torch.cat([entry[2] for entry in res], 0).numpy()

    total_genders = torch.cat([entry[3] for entry in res], 0).numpy()

    total_img_features = torch.cat([entry[4] for entry in res], 0).numpy()

    out_1 = list()
    out_2 = list()

    for (preds, preds_scores, targets, genders, img_features, image_ids) in zip(total_preds, total_preds_scores, total_targets, total_genders,
                                                               total_img_features, total_img_ids):
      out_1.append({'img_id':image_ids, 'gender':genders, 'targets':targets, 'preds':preds, 'scores':preds_scores, 'img_feature':img_features})
      out_2.append({'img_id':image_ids, 'gender':genders, 'targets':targets, 'scores':preds_scores,'preds':preds})
    
    return out_1, out_2

# Data samplers
val_data_2 = CocoObjectGender(args, annotation_dir = args.annotation_dir, \
        image_dir = args.image_dir, split = 'val', transform = None)

val_loader_2 = torch.utils.data.DataLoader(val_data_2, batch_size = args.batch_size,
            shuffle = False, num_workers = 2, pin_memory = True)

val_out_1, val_out_2 = get_embeddings(args, model2, val_loader_2, device)

val_out_name = 'val_embeddings.data'
val_out_name_2 = 'val.data'

with open(os.path.join(args.save_dir, val_out_name),'wb') as outfile:
  pickle.dump(val_out_1,outfile)

with open(os.path.join(args.save_dir, val_out_name_2),'wb') as outfile:
  pickle.dump(val_out_2,outfile)