import pandas as pd 
import numpy as np 
import os
import pickle
import argparse

from collections import Counter

from nltk.tokenize import  word_tokenize
from collections import defaultdict

from matplotlib import pyplot as plt

from sklearn.cluster import KMeans_logan
from sklearn.cluster import KMeans
from sklearn import metrics

import utils


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default = './data', help='data directory')
parser.add_argument('--results_dir', default = './results', help='results directory for saving images')
parser.add_argument('--test_pred', default = 'val.data', help='File name of test set with predictions and embeddings')

parser.add_argument('--save_clusters',action='store_true')

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--lamb', type=int, default=5)

args = parser.parse_args()

np.random.seed(args.seed)


"""Loading and formatting the data"""

samples = pickle.load(open(os.path.join(args.data_dir, args.test_pred),'rb'))

id2object = {0: 'toilet', 1: 'teddy_bear', 2: 'sports_ball', 3: 'bicycle', 4: 'kite', 5: 'skis', 6: 'tennis_racket', 7: 'donut', 8: 'snowboard', 9: 'sandwich', 10: 'motorcycle', 11: 'oven', 12: 'keyboard', 13: 'scissors', 14: 'chair', 15: 'couch', 16: 'mouse', 17: 'clock', 18: 'boat', 19: 'apple', 20: 'sheep', 21: 'horse', 22: 'giraffe', 23: 'tv', 24: 'stop_sign', 25: 'toaster', 26: 'bowl', 27: 'microwave', 28: 'bench', 29: 'fire_hydrant', 30: 'book', 31: 'elephant', 32: 'orange', 33: 'tie', 34: 'banana', 35: 'knife', 36: 'pizza', 37: 'fork', 38: 'hair_drier', 39: 'frisbee', 40: 'umbrella', 41: 'bottle', 42: 'bus', 43: 'zebra', 44: 'bear', 45: 'vase', 46: 'toothbrush', 47: 'spoon', 48: 'train', 49: 'airplane', 50: 'potted_plant', 51: 'handbag', 52: 'cell_phone', 53: 'traffic_light', 54: 'bird', 55: 'broccoli', 56: 'refrigerator', 57: 'laptop', 58: 'remote', 59: 'surfboard', 60: 'cow', 61: 'dining_table', 62: 'hot_dog', 63: 'car', 64: 'cup', 65: 'skateboard', 66: 'dog', 67: 'bed', 68: 'cat', 69: 'baseball_glove', 70: 'carrot', 71: 'truck', 72: 'parking_meter', 73: 'suitcase', 74: 'cake', 75: 'wine_glass', 76: 'baseball_bat', 77: 'backpack', 78: 'sink'}

num_objects = len(id2object)

obj_dict = {}

embeds = []
genders = []

for i in range(num_objects):
  obj_dict[i] = {}
  obj_dict[i]['target']=[]
  obj_dict[i]['pred']=[]

for sample in samples:
  if(sample['gender'][0]==1 or sample['gender'][1]==1):
    embeds.append(sample['img_feature'])
    genders.append(sample['gender'][0])
    for i in range(num_objects):  
      obj_dict[i]['target'].append(sample['targets'][i])
      obj_dict[i]['pred'].append(sample['preds'][i])

obj_df = pd.DataFrame({'gender':genders})

for i in range(num_objects):
  obj_df[f'pred_{i}']=obj_dict[i]['pred']
  obj_df[f'target_{i}']=obj_dict[i]['target']

obj_df.head(10)


ta, tb = 'male', 'female'
ta_df = obj_df[(obj_df['gender'] == 1)]
tb_df = obj_df[(obj_df['gender'] == 0)]

print(f"{len(ta_df)} {ta} examples, {len(tb_df)} {tb} examples")
print(f"{len(obj_df)} total examples")

num_bias = 0
for i in range(num_objects):
  print('For object \''+id2object[i]+'\': ')
  print(f"total samples = {len(obj_df[obj_df[f'target_{i}'] == 1])}")
  ta_acc = len(ta_df[ta_df[f'pred_{i}'] == ta_df[f'target_{i}']])/len(ta_df)
  tb_acc = len(tb_df[tb_df[f'pred_{i}'] == tb_df[f'target_{i}']])/len(tb_df)
  print(f"acc {ta}:{ta_acc}, {tb}:{tb_acc}, diff:{ta_acc - tb_acc:.4f}")
  print(f"overall acc:{len(obj_df[obj_df[f'pred_{i}'] == obj_df[f'target_{i}']])/len(obj_df)}")
  print()
  if((ta_acc - tb_acc)>=0.05):
    num_bias+=1

print(f"Number of samples with group bias greater than 5% = {num_bias}")

tgs = np.asarray(embeds)[np.concatenate((ta_df.index.values, tb_df.index.values))]
print(tgs.shape)

# get all the prediction results regarding to these 2 groups
tgt = pd.concat([ta_df, tb_df], axis = 0)



"""Clustering using K-means and computing local biases in the clusters."""


NCluster = 10
kmeans = KMeans(NCluster, random_state=10, n_jobs=20, precompute_distances = False, algorithm = 'full').fit(tgs)
kmeans_clusters_raw = kmeans.labels_
kcenters = kmeans.cluster_centers_

#calculate male/female instances in each cluster
gender_c = []
for i in range(NCluster):
    idxc = np.where(kmeans_clusters_raw == i)
    tgtc = tgt.iloc[idxc]
    gender_c.append(np.array([len(tgtc[tgtc['gender'] == 1]), len(tgtc[tgtc['gender'] == 0])]))

g, NCluster, kmeans_clusters = utils.merge_clusters(gender_c, kcenters, kmeans_clusters_raw, tgs)

res4plot = []
num_bias = 0
for j in range(num_objects):
  print('For object \''+id2object[j]+'\': ')
  max_diff = 0
  mi, mtac, mtbc, maccac, maccbc = 0, 0, 0, 0, 0
  for i in np.unique(kmeans_clusters):
    idxc = np.where(kmeans_clusters == i)
    tgtc = tgt.iloc[idxc]
    tac = tgtc[tgtc['gender'] == 1]
    tbc = tgtc[tgtc['gender'] == 0]
    accac = metrics.accuracy_score(y_true=tac[f'target_{j}'], y_pred = tac[f'pred_{j}'])
    accbc = metrics.accuracy_score(y_true=tbc[f'target_{j}'], y_pred = tbc[f'pred_{j}'])
    res4plot.append([accac, accbc, abs(accac-accbc)])
    if (max_diff<abs(accac-accbc)):
      mi, mtac, mtbc, maccac, maccbc, max_diff = i, len(tac), len(tbc), accac, accbc, abs(accac-accbc) 
  print(f"c {mi}, {mtac} {ta}, {mtbc} {tb}, acc:{maccac:.5f}, {maccbc: .5f}, acc_d:{maccac - maccbc:.4f}")
  if(max_diff>=0.05):
    num_bias+=1
  print()
print(f"Number of samples with local group bias greater than 5% = {num_bias}")



"""Clustering using LOGAN. For this part, you need to place the files in the ./clusters folder to clusters folder in your sklearn directory."""


if args.save_clusters and (not os.path.exists(args.results_dir)):
  os.makedirs(args.results_dir)

cluster_dict = {}
num_bias = 0
for j in range(num_objects):
  print('For object \''+id2object[j]+'\': ')
  attributes = np.stack((tgt[f'target_{j}'], tgt[f'pred_{j}'], atts), axis = -1).astype("float32")

  NCluster = 10
  ikmeans = KMeans_logan(NCluster, random_state=10, n_jobs = -1, precompute_distances = False, eta= -args.lamb, algorithm = 'full',n_init=1).fit(tgs, attributes)
  ikmeans_clusters_raw = ikmeans.labels_
  ikcenters = ikmeans.cluster_centers_

  #calculate male/female instances in each cluster
  igender_c = []
  for i in range(NCluster):
      idxc = np.where(ikmeans_clusters_raw == i)
      tgtc = tgt.iloc[idxc]
      igender_c.append(np.array([len(tgtc[tgtc['gender'] == 1]), len(tgtc[tgtc['gender'] == 0])]))

  #Merge small clusters
  ig, iNCluster, ikmeans_clusters = utils.merge_clusters(igender_c, ikcenters, ikmeans_clusters_raw, tgs)

  # Save for future use
  cluster_dict[j] = (iNCluster, ikmeans_clusters)
  
  max_diff = 0
  mi, mtac, mtbc, maccac, maccbc = 0, 0, 0, 0, 0
  for i in range(iNCluster):
    idxc = np.where(ikmeans_clusters == i)
    tgtc = tgt.iloc[idxc]
    tac = tgtc[tgtc['gender'] == 1]
    tbc = tgtc[tgtc['gender'] == 0]
    accac = metrics.accuracy_score(y_true=tac[f'target_{j}'], y_pred = tac[f'pred_{j}'])
    accbc = metrics.accuracy_score(y_true=tbc[f'target_{j}'], y_pred = tbc[f'pred_{j}'])
    if (max_diff<abs(accac-accbc)):
      mi, mtac, mtbc, maccac, maccbc, max_diff = i, len(tac), len(tbc), accac, accbc, abs(accac-accbc) 
  print(f"c {mi}, {mtac} {ta}, {mtbc} {tb}, acc:{maccac:.5f}, {maccbc: .5f}, acc_d:{maccac - maccbc:.4f}")
  if(max_diff>=0.05):
    num_bias+=1
  print()

print(f"Number of samples with local group bias greater than 5% = {num_bias}")

if args.save_clusters:
  with open(os.path.join(args.results_dir, 'clusters.bin'),'wb') as outfile:
    pickle.dump(cluster_dict,outfile)