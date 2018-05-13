import sys
import json
import tensorflow as tf
import time
from compare import load_and_align_data


graph_dir = "../data/models/20180402-114759/"
graph_meta = graph_dir + "model-20180402-114759.meta"
graph_ckpt = graph_dir + "model-20180402-114759.ckpt-275.data-00000-of-00001"


def image_pipeline(imgf, target_dims=[160, 160, 3]):
    """This function encapsulates process from image file path load through
     input into the TensorFlow model/graph."""
    # args are: list of image paths, image size, margin (padding), and gpu mem
    return load_and_align_data([imgf], target_dims[0], 20,  0.0)


def get_graph_infer(metafile, ckptfile):
    """This returns a function that will perform inference using the graph
    and session state instantiated from the passed files."""
    # loading w/this instead of with convention to make graph persist
    # for closure
    tf.Graph().as_default()
    sess = tf.Session()
    # restore model from files
    saver = tf.train.import_meta_graph(metafile)
    saver.restore(sess, ckptfile)

    # match placeholders in tf graph with inputs/outputs
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    # nested model inference function we'll return to do future inferences
    # from graph
    def infer(imgf):
        feed_dict = {images_placeholder: image_pipeline(imgf),
                     phase_train: False}
        start = time.time()
        embedding_result = sess.run(embeddings, feed_dict=feed_dict)
        end = time.time()
        print(end - start)
        return embedding_result

    return infer

if __name__ == "__main__":
    imgf = sys.argv[1]
    my_infer = get_graph_infer(graph_meta, graph_ckpt)
    my_embedding = my_infer(imgf)
    print(json.dumps(my_embedding.tolist()))
