
import os
import tensorflow as tf


def get_latest_check_num(base_directory):
    glob = os.path.join(base_directory, '*index')
    # glob = os.path.join(base_directory, '*')
    def extract_iteration(x):
        x = x[:x.rfind('.')]
        return int(x[x.rfind('-') + 1:])
    try:
        checkpoint_files = tf.gfile.Glob(glob)
    except tf.errors.NotFoundError:
        return -1
    try:
        latest_iteration = max(extract_iteration(x) for x in checkpoint_files)
        return latest_iteration
    except ValueError:
        return -1