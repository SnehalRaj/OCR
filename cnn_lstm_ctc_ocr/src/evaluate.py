# CNN-LSTM-CTC-OCR
# Copyright (C) 2017,2018 Jerod Weinman, Abyaya Lamsal, Benjamin Gafford
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


# evaluate.py -- Streams evaluation statistics (i.e., character error
#   rate, sequence error rate) for a single batch whenever a new model
#   checkpoint appears

import os
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import six
import model_fn
import pipeline
import filters

# Filters out information to just show a stream of results
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_integer('batch_size', 2**9,
                            """Eval batch size""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 120,
                            """Time between test runs""")

tf.app.flags.DEFINE_string('model', '../data/model',
                           """Directory for event logs and checkpoints""")
tf.app.flags.DEFINE_string('output', 'test',
                           """Sub-directory of model for test summary events""")

tf.app.flags.DEFINE_string('test_path', '../data/',
                           """Base directory for test/validation data""")
tf.app.flags.DEFINE_string('filename_pattern', 'val/words-*',
                           """File pattern for input data""")
tf.app.flags.DEFINE_integer('num_input_threads', 4,
                            """Number of readers for input data""")

tf.app.flags.DEFINE_integer('min_image_width', None,
                            """Minimum allowable input image width""")
tf.app.flags.DEFINE_integer('max_image_width', None,
                            """Maximum allowable input image width""")
tf.app.flags.DEFINE_integer('min_string_length', None,
                            """Minimum allowable input string length""")
tf.app.flags.DEFINE_integer('max_string_length', None,
                            """Maximum allowable input string_length""")

tf.app.flags.DEFINE_boolean('bucket_data', False,
                            """Bucket training data by width for efficiency""")


def _get_input():
    """
    Get tf.data.Dataset object according to command-line flags for evaluation
    using tf.estimator.Estimator

    Note: Default behavior is bucketing according to default bucket boundaries
    listed in pipeline.get_data

    Returns:
      features, labels
                feature structure can be seen in postbatch_fn 
                in mjsynth.py or maptextsynth.py for static or dynamic
                data pipelines respectively
    """

    # WARNING: More than two filters causes SEVERE throughput slowdown
    filter_fn = filters.input_filter_fn(min_image_width=FLAGS.min_image_width,
                                        max_image_width=FLAGS.max_image_width,
                                        min_string_length=FLAGS.min_string_length,
                                        max_string_length=FLAGS.max_string_length)

    # Pack keyword arguments into dictionary
    data_args = {'base_dir': FLAGS.test_path,
                 'file_patterns': str.split(FLAGS.filename_pattern, ','),
                 'num_threads': FLAGS.num_input_threads,
                 'batch_size': FLAGS.batch_size,
                 'filter_fn': filter_fn
                 }

    if not FLAGS.bucket_data:  # Turn off bucketing (on by default in pipeline)
        data_args['boundaries'] = None

    # Get data according to flags
    dataset = pipeline.get_data(use_static_data=True, **data_args)

    return dataset


# Taken from the official source code of Tensorflow
# Licensed under the Apache License, Version 2.0
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/estimator/estimator.py
def _extract_metric_update_ops(eval_dict):
    """Separate update operations from metric value operations."""
    update_ops = []
    value_ops = {}
    # Sort metrics lexicographically so graph is identical every time.
    for name, metric_ops in sorted(six.iteritems(eval_dict)):
        value_ops[name] = metric_ops[0]
        update_ops.append(metric_ops[1])

    if update_ops:
        update_op = control_flow_ops.group(*update_ops)
    else:
        update_op = None

    return update_op, value_ops


def _get_config():
    """Setup session config to soften device placement"""
    device_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)

    return device_config


def main(argv=None):

    dataset = _get_input()

    # Extract input tensors for evaluation
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    # Construct the evaluation function
    evaluate_fn = model_fn.evaluate_fn()

    # Wrap the ops in an Estimator spec object
    estimator_spec = evaluate_fn(features, labels,
                                 tf.estimator.ModeKeys.EVAL,
                                 {'continuous_eval': True})

    # Extract the necessary ops and the final tensors from the estimator spec
    update_op, value_ops = _extract_metric_update_ops(
        estimator_spec.eval_metric_ops)

    # Specify to evaluate N number of batches (in this case N==1)
    stop_hook = tf.contrib.training.StopAfterNEvalsHook(1)

    # Create summaries of values added to tf.GraphKeys.SUMMARIES
    summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.model,
                                                        FLAGS.output))
    summary_hook = tf.contrib.training.SummaryAtEndHook(
        summary_writer=summary_writer)

    # Evaluate repeatedly once a new checkpoint is found
    tf.contrib.training.evaluate_repeatedly(
        checkpoint_dir=FLAGS.model, eval_ops=update_op, final_ops=value_ops,
        hooks=[stop_hook, summary_hook], config=_get_config(),
        eval_interval_secs=FLAGS.eval_interval_secs)


if __name__ == '__main__':
    tf.app.run()
