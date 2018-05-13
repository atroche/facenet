# coding: utf-8
import numpy as np
import scipy.spatial.distance
import os

e = np.load("zuck_embeddings.npy")

m = np.average(e, axis=0)

eu = scipy.spatial.distance.euclidean

distances = [eu(m, emb) for emb in e]
files = os.listdir("datasets/zuck_aligned/Mark_Zuckerberg/")

weirdest_zucks = sorted(zip(files, distances), key=lambda x: x[1])

