import pandas as pd
import scipy
from scipy.spatial import distance_matrix
import networkx as nx
import numpy as np
from scipy.stats import describe
from PIL import Image, ImageDraw
import cv2


def getGreedyPerm(D, N):
    """
    A Naive O(N^2) algorithm to do furthest points sampling
    Parameters
    ----------
    D : ndarray (N, N) 
        An NxN distance matrix for points
    Return
    ------
    tuple (list, list) 
        (permutation (N-length array of indices), 
        lambdas (N-length array of insertion radii))
    """
    
    #By default, takes the first point in the list to be the
    #first point in the permutation, but could be random
    perm = np.zeros(N, dtype=np.int64)
    lambdas = np.zeros(N)
    ds = D[0, :]
    for i in range(1, N):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, D[idx, :])
    return (perm, lambdas)




centroid = pd.read_csv("TCGA-2Z-A9J9-01A-01-TS1_nuc-centroids.csv")
dis_mat = scipy.spatial.distance_matrix(centroid, centroid)
FPS, _ = getGreedyPerm(dis_mat, 250)
FPS = list(set(FPS))
thres = 100
F_dismat = dis_mat[FPS, :][:, FPS]
r,c =  np.where(F_dismat<thres)


width = 1000
height = 1000
img = np.zeros((height, width, 3), np.uint8)
img = cv2.imread("../MoNuSegTestData/TCGA-2Z-A9J9-01A-01-TS1.tif")
for i in range(r.shape[0]):
    img = cv2.line(img, (int(centroid.iloc[[FPS[r[i]]]]['x']), int(centroid.iloc[[FPS[r[i]]]]['y'])), 
    (int(centroid.iloc[[FPS[c[i]]]]['x']), int(centroid.iloc[[FPS[c[i]]]]['y'])), (0,255,0), 1)


cv2.imwrite('test.png',img)

