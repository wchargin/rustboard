from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf


def main():
    tf.compat.v1.enable_eager_execution()
    base_logdir = "test_data"
    if tf.io.gfile.exists(base_logdir):
        raise ValueError("logdir %r exists; please remove it" % base_logdir)
    for run_idx in range(4):
        logdir = os.path.join(base_logdir, "run_%03d" % run_idx)
        writer = tf.compat.v2.summary.create_file_writer(logdir)
        with writer.as_default():
            generate_run_data(run_idx)


def generate_run_data(run_idx):
    for step in range(100):
        accuracy = 1.0 - 1.0 / ((run_idx + 1) * (step + 1))
        loss = 3 * (1 - accuracy) ** 2
        tf.compat.v2.summary.scalar("accuracy", accuracy, step=step)
        tf.compat.v2.summary.scalar("loss", loss, step=step)


if __name__ == "__main__":
    main()
