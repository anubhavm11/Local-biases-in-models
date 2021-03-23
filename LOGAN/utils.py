import numpy as np
from copy import deepcopy

def merge_clusters(gender_c, kmeans_c, kmeans_clusters, features):
    ng = deepcopy(gender_c)
    kcc = deepcopy(kmeans_c)
    n2o = list(range(len(gender_c)))
    kc = deepcopy(kmeans_clusters)
    while True:
        min_c = np.min(np.array(ng))
        if min_c >= 20:
            print("Done merge for all clusters have at least 20 M/F images")
            break
        if len(ng) <= 5:
            print("Finish merging as only a few clusters left")
            break
        for c in range(len(ng)):
            if ng[c][0] < 20 or ng[c][1] < 20: #<10 m/f examples => merge to the closet cluster
                distances = np.dot(np.delete(kcc, c, 0), kcc[c])
                merge2 = np.argmin(distances)
                if merge2 >= c:
                    merge2 += 1
#                 print(c, merge2)
                ng[merge2] += ng[c]
                ng[c] = ng[merge2]
                kc[np.where(kc == n2o[c])] = n2o[merge2]
#                 print(kc)
                kcc[merge2] = np.mean(features[np.where(kc == n2o[merge2])], axis = 0)
                ng.pop(c)
                kcc = np.delete(kcc, c, 0)
#                 print(ng, len(ng), len(kcc), np.unique(kc))
                n2o.pop(c)
                break
    #remap the clusters to avoid skipped values
    oc2nc = {}
    for idx in range(len(np.unique(kc))):
        oc2nc[np.unique(kc)[idx]] = idx
    kc = [oc2nc[x] for x in kc]
    return ng, len(np.unique(kc)), np.array(kc)

def get_inertia(clusters, embs):
    dis = 0
    for c in np.unique(clusters):
        idxc = np.where(clusters == c)
        centers = np.average(embs[idxc], axis = 0)
        norms =  np.linalg.norm((embs[idxc]- centers), axis = 0)**2
        dis += norms.sum()
    return dis