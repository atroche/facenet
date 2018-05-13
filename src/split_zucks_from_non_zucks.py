import time
import os
from shutil import copyfile
import glob
import itertools
import pickle

import numpy as np
from PIL import Image

import scipy.spatial.distance

cosine = scipy.spatial.distance.cosine
eu = scipy.spatial.distance.euclidean

dir = "datasets/videos/aligned/bloomberg_docco/MZ"

paths_and_embeddings = pickle.load(open("bloomberg_docco_embeddings.p", "rb"))

known_zuck = [pe for pe in paths_and_embeddings if pe[0] == dir + "frame_60.0_face_0.png"][0]

# [Image.open(open(path, "rb")).show() for path, embedding in paths_and_embeddings[:10]]


paths_sorted = sorted([(pe[0], cosine(known_zuck[1], pe[1])) for pe in paths_and_embeddings],
       key=lambda x:x[1])

paths_sorted_eu = sorted([(pe[0], eu(known_zuck[1], pe[1])) for pe in paths_and_embeddings],
                      key=lambda x:x[1])


for index, (path, embedding) in enumerate(paths_sorted):
    copyfile(path, "datasets/videos/faces/sorted/F8_conf_2018/%s.png" % index)

sorted_paths = pickle.load(open("sorted_how_paths.p", "rb"))

sorted_paths[0]

for index, path in enumerate(sorted_paths):
    copyfile(path, "datasets/videos/faces/sorted/how_to/%s.jpg" % index)

