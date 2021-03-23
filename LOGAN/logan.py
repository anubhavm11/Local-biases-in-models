import pandas as pd 
import numpy as np 
import os

from collections import Counter

from nltk.tokenize import  word_tokenize
from collections import defaultdict
from copy import deepcopy

from matplotlib import pyplot as plt
import argparse

from sklearn.cluster import KMeans_logan
from sklearn.cluster import KMeans
from sklearn import metrics

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default = './data', help='data directory')
parser.add_argument('--results_dir', default = './results', help='results directory for saving images')

parser.add_argument('--test_pred', default = 'pred_test_bw_withpredscores.csv', help='File name of test set with predictions')
parser.add_argument('--test_embed', default = 'second2last_mean.npy', help='Numpy file name of embeddings for the test set')
parser.add_argument('--demographic', default = 'gender', help='Demographic for computing bias scores')

parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--lamb', type=int, default=5)

args = parser.parse_args()

np.random.seed(args.seed)


"""Loading test predictions and embeddings. We only look at examples with male/female or black/white labels."""

test_file = os.path.join(args.data_dir, args.test_pred)
res = pd.read_csv(test_file, index_col=0)

test_embs = np.load(os.path.join(args.data_dir, args.test_embed)).astype('float32')
print(f"test embedding shape:{test_embs.shape}")

if args.demographic == 'gender':
    ta, tb = 'male', 'female'
else:
    ta, tb = 'black', 'white'

ta_df = res[(res[ta] >= 0.5) & (res[tb] < 0.5)]
tb_df = res[(res[tb] >= 0.5) & (res[ta] < 0.5)]
print(f"{len(ta_df)} {ta} examples, {len(tb_df)} {tb} examples")
print(f"{len(res)} total examples")

prediction_column = 'predictions'
ta_acc = len(ta_df[ta_df[prediction_column] == ta_df['target']])/len(ta_df)
tb_acc = len(tb_df[tb_df[prediction_column] == tb_df['target']])/len(tb_df)
print(f"acc {ta}:{ta_acc}, {tb}:{tb_acc}, diff:{ta_acc - tb_acc:.4f}")
print(f"overall acc:{len(res[res[prediction_column] == res['target']])/len(res)}")

tgs = test_embs[np.concatenate((ta_df.index.values, tb_df.index.values))]
print(tgs.shape)

print(np.concatenate((ta_df.index.values, tb_df.index.values)))

# get all the prediction results regarding to these 2 groups
tgt = pd.concat([ta_df, tb_df], axis = 0)

print(f"for "+args.demographic+" group, the overall accuracy is {metrics.accuracy_score(y_true = tgt['target'], y_pred = tgt[prediction_column])}")



"""Clustering using K-means and computing local biases in the clusters."""

if(not os.path.exists(args.results_dir)):
  os.makedirs(args.results_dir)

NCluster = 10
kmeans = KMeans(NCluster, random_state=10, n_jobs=20, precompute_distances = False, algorithm = 'full').fit(tgs)
kmeans_clusters_raw = kmeans.labels_
kcenters = kmeans.cluster_centers_

#calculate #white/black instances in each cluster
gender_c = []
for i in range(NCluster):
    idxc = np.where(kmeans_clusters_raw == i)
    tgtc = tgt.iloc[idxc]
    gender_c.append(np.array([len(tgtc[tgtc[ta] >= 0.5]), len(tgtc[tgtc[tb] >= 0.5])]))


g, NCluster, kmeans_clusters = utils.merge_clusters(gender_c, kcenters, kmeans_clusters_raw, tgs)

res4plot = []
for i in np.unique(kmeans_clusters):
    idxc = np.where(kmeans_clusters == i)
    tgtc = tgt.iloc[idxc]
    tac = tgtc[tgtc[ta] >= 0.5]
    tbc = tgtc[tgtc[tb] >= 0.5]
    accac = metrics.accuracy_score(y_true=tac['target'], y_pred = tac[prediction_column])
    accbc = metrics.accuracy_score(y_true=tbc['target'], y_pred = tbc[prediction_column])
    res4plot.append([accac, accbc, abs(accac-accbc)])
    print(f"c {i}, {len(tac)} {ta}, {len(tbc)} {tb}, acc:{accac:.5f}, {accbc: .5f}, acc_d:{accac - accbc:.4f}")

plt.close('all')
plt.rc('font', size=13)  
res4plot.append([ta_acc, tb_acc, abs(ta_acc - tb_acc)])
for i in range(len(res4plot)):
    x = [i, i]
    y = [res4plot[i][0], res4plot[i][1]]
    plt.plot(x, y, '--', color ='green', linewidth=2)
plt.plot([8, 8], [ta_acc, tb_acc], '--', color ='green', linewidth=2)
plt.scatter(range(len(res4plot) ),np.array(res4plot)[:,0], label = 'M', marker ='o', s=80)
plt.scatter(range(len(res4plot)), np.array(res4plot)[:, 1], label = 'F', marker = 's', s=80)
plt.ylabel('Accuracy')
# plt.ylim([0.48, 0.95])
plt.xticks(range(len(res4plot) -1))
plt.legend()
plt.savefig(os.path.join(args.results_dir,args.demographic+'_acc_gap_kmeans.png'))
plt.show()



"""Clustering using LOGAN. For this part, you need to place the files in the ./clusters folder to clusters folder in your sklearn directory."""


atts = np.concatenate((np.zeros(len(ta_df)), np.ones(len(tb_df))))
attributes = np.stack((tgt['target'], tgt[prediction_column], atts), axis = -1).astype("float32")
# print(attributes)


NCluster = 10
ikmeans = KMeans_logan(NCluster, random_state=10, n_jobs = -1, precompute_distances = False, eta= -args.lamb, algorithm = 'full',n_init=1).fit(tgs, attributes)
ikmeans_clusters_raw = ikmeans.labels_
ikcenters = ikmeans.cluster_centers_

#calculate #white/black or male/female instances in each cluster
igender_c = []
for i in range(NCluster):
    idxc = np.where(ikmeans_clusters_raw == i)
    tgtc = tgt.iloc[idxc]
    igender_c.append(np.array([len(tgtc[tgtc[ta] >= 0.5]), len(tgtc[tgtc[tb] >= 0.5])]))

ig, iNCluster, ikmeans_clusters = utils.merge_clusters(igender_c, ikcenters, ikmeans_clusters_raw, tgs)

res4plot = []
for i in range(iNCluster):
    idxc = np.where(ikmeans_clusters == i)
    tgtc = tgt.iloc[idxc]
    tac = tgtc[tgtc[ta] >= 0.5]
    tbc = tgtc[tgtc[tb] >= 0.5]
    accac = metrics.accuracy_score(y_true=tac['target'], y_pred = tac[prediction_column])
    accbc = metrics.accuracy_score(y_true=tbc['target'], y_pred = tbc[prediction_column])
    res4plot.append([accac, accbc, abs(accac - accbc)])
    print(f"c {i}, {len(tac)} {ta}, {len(tbc)} {tb}, acc:{accac:.5f}, {accbc: .5f}, acc_d:{accac - accbc:.4f}")

inertia_kmeans = get_inertia(kmeans_clusters, tgs)
inertia_ikmeans = get_inertia(ikmeans_clusters, tgs)
print(f"inertia: kmeans{inertia_kmeans}, ikmeans:{inertia_ikmeans}, ratio:{inertia_ikmeans / inertia_kmeans}")

plt.close('all')
plt.rc('font', size=13)  
res4plot.append([ta_acc, tb_acc, abs(ta_acc - tb_acc)])
for i in range(len(res4plot)):
    x = [i, i]
    y = [res4plot[i][0], res4plot[i][1]]
    plt.plot(x, y, '--', color ='green', linewidth=2)
plt.plot([8, 8], [ta_acc, tb_acc], '--', color ='green', linewidth=2)
plt.scatter(range(len(res4plot) ),np.array(res4plot)[:,0], label = 'M', marker ='o', s=80)
plt.scatter(range(len(res4plot)), np.array(res4plot)[:, 1], label = 'F', marker = 's', s=80)
plt.ylabel('Accuracy')
# plt.ylim([0.48, 0.95])
plt.xticks(range(len(res4plot) -1))
plt.legend()
plt.savefig(os.path.join(args.results_dir,args.demographic + f'_acc_gap_logan_lamb={args.lamb}.png'))
plt.show()
