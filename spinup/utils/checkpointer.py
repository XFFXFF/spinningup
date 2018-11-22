
import os
import sys
import tensorflow as tf

from spinup.utils.logx import colorize


def get_latest_check_num(base_directory):
    glob = os.path.join(base_directory, '*index')
    # glob = os.path.join(base_directory, '*')
    def extract_iteration(x):
        x = x[:x.rfind('.')]
        return int(x[x.rfind('-') + 1:])
    try:
        checkpoint_files = tf.gfile.Glob(glob)
    except tf.errors.NotFoundError:
        print(colorize('There is no trained model that meet you requirements', 'red', bold=True))
        sys.exit(0)
    try:
        latest_iteration = max(extract_iteration(x) for x in checkpoint_files)
        return latest_iteration
    except ValueError:
        return -1