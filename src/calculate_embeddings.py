import time
import glob
import pickle

import numpy as np
import scipy.misc
import tensorflow as tf

import facenet

image_size = (160, 160)


model_dir = "facenet/data/models/20180402-114759/"
graph_meta = model_dir + "model-20180402-114759.meta"
graph_ckpt = model_dir + "model-20180402-114759.ckpt-275.data-00000-of-00001"
#
def read_image(path):
    return scipy.misc.imread(path, mode='RGB')


def forward_pass(paths):
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    images = [read_image(path) for path in paths]

    feed_dict = {images_placeholder: np.stack(images),
                 phase_train: False}
    start = time.time()
    embedding_result = sess.run(embeddings, feed_dict=feed_dict)
    end = time.time()
    print("took time:")
    print(end - start)
    return list(zip(paths, embedding_result))

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


# lfw_dir = "datasets/lfw_mtcnnpy_160"

def go(image_path, pickle_filename):
    facenet.load_model(model_dir)

    paths = glob.glob(image_path + "/**.png") + glob.glob(image_path + "/**.jpg")
    paths_and_embeddings = []
    for chunk_num, paths in enumerate(chunks(paths, 300)):
        print("chunk num:")
        print(chunk_num)

        paths_and_embeddings += forward_pass(paths)

    print(len(paths_and_embeddings))
    with open(pickle_filename, "wb") as f:
        pickle.dump(paths_and_embeddings, f)

with tf.Graph().as_default():
    with tf.Session() as sess:
        go("datasets/new_non_zucks/**", "non_zucks_with_how_to.p")
# Chunked version (ran into memory issues when I did 15000 images at once)

# with tf.Graph().as_default():
#     with tf.Session() as sess:
#         facenet.load_model(model_dir)
#
#         for chunk_num, paths in enumerate(chunks(paths, 300)):
#             print("chunk num:")
#             print(chunk_num)
#             paths_and_embeddings = forward_pass(paths)
#             with open("lfw_embeddings_%s.p" % chunk_num, "wb") as f:
#                 pickle.dump(paths_and_embeddings, f)


