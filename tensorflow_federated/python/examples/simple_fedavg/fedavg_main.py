# Copyright 2020, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Simple FedAvg to train EMNIST.

This is intended to be a minimal stand-alone experiment script built on top of
core TFF.
"""
import collections
import time

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
import os

import tensorflow_federated as tff
from tensorflow_federated.python.examples.simple_fedavg import dataset_shakespeare
from tensorflow_federated.python.examples.simple_fedavg import keras_metrics
from tensorflow_federated.python.examples.simple_fedavg import models
from tensorflow_federated.python.examples.simple_fedavg import simple_fedavg_tf
from tensorflow_federated.python.examples.simple_fedavg import simple_fedavg_tff
from tensorflow_federated.python.examples.simple_fedavg.dataset_cifar10 import get_cifar10_federated_datasets
from tensorflow_federated.python.examples.simple_fedavg.dataset_femnist import get_emnist_dataset
from tensorflow_federated.python.examples.simple_fedavg.dataset_shakespeare import construct_character_level_datasets
from tensorflow_federated.python.examples.simple_fedavg.resnet_models import create_resnet18

# Training hyperparameters
flags.DEFINE_integer('total_rounds', 256, 'Number of total training rounds.')
flags.DEFINE_integer('rounds_per_eval', 1, 'How often to evaluate')
flags.DEFINE_integer('train_clients_per_round', 2,
                     'How many clients to sample per round.')
flags.DEFINE_integer('client_epochs_per_round', 1,
                     'Number of epochs in the client to take per round.')
flags.DEFINE_integer('batch_size', 64, 'Batch size used on the client.')
flags.DEFINE_integer('test_batch_size', 64, 'Minibatch size of test data.')
flags.DEFINE_integer('seed', 0, 'random seed.')
flags.DEFINE_string('data_dir', "", 'customized data directory')
flags.DEFINE_string('dataset', "", 'dataset name')
flags.DEFINE_boolean('test_all', False, 'test all clients')
flags.DEFINE_boolean('test_in_server', False, 'test in centralized server')
flags.DEFINE_integer('gpu', -1, '-1 means any number of gpu available')

# Optimizer configuration (this defines one or more flags per optimizer).
flags.DEFINE_float('server_learning_rate', 1.0, 'Server learning rate.')
flags.DEFINE_float('client_learning_rate', 0.1, 'Client learning rate.')

FLAGS = flags.FLAGS

DATASET_FEMNIST = "femnist"
DATASET_SHAKESPEARE = "shakespeare"
DATASET_CIFAR10 = "cifar10"

VOCAB_SIZE = len(dataset_shakespeare.CHAR_VOCAB) + 4

CIFAR_SHAPE = (32, 32, 3)
CROP_SHAPE = (32, 32, 3)
NUM_CLASSES = 10


def server_optimizer_fn():
    return tf.keras.optimizers.SGD(learning_rate=FLAGS.server_learning_rate)


def client_optimizer_fn():
    # return tf.keras.optimizers.Adam(learning_rate=FLAGS.client_learning_rate)
    return tf.keras.optimizers.SGD(learning_rate=FLAGS.client_learning_rate, momentum=0.9)


def get_dataset():
    train_data, test_data = None, None
    if FLAGS.dataset == DATASET_FEMNIST:
        train_data, test_data = get_emnist_dataset(FLAGS)
    elif FLAGS.dataset == DATASET_SHAKESPEARE:
        train_data, test_data = construct_character_level_datasets(FLAGS.data_dir,
                                                                   FLAGS.batch_size,
                                                                   FLAGS.client_epochs_per_round,
                                                                   FLAGS.test_batch_size)
    elif FLAGS.dataset == DATASET_CIFAR10:
        train_data, test_data = get_cifar10_federated_datasets(FLAGS.data_dir,
                                                               FLAGS.batch_size,
                                                               FLAGS.test_batch_size,
                                                               FLAGS.client_epochs_per_round,
                                                               crop_shape=CROP_SHAPE)
    return train_data, test_data


def get_model_fn():
    def tff_emnist_model_fn():
        """Constructs a fully initialized model for use in federated averaging."""
        keras_model = models.create_original_fedavg_cnn_model(only_digits=False)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        element_spec = collections.OrderedDict(
            x=tf.TensorSpec([None, 28, 28, 1], tf.float32),
            y=tf.TensorSpec([None], tf.int32))
        return simple_fedavg_tf.KerasModelWrapper(keras_model, element_spec, loss)
        # return simple_fedavg_tf.KerasModelWrapper(keras_model, test_data.element_spec, loss)

    def tff_shakespeare_model_fn():
        """Constructs a fully initialized model for use in federated averaging."""
        keras_model = models.create_recurrent_model(VOCAB_SIZE, dataset_shakespeare.SEQUENCE_LENGTH)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        element_spec = collections.OrderedDict(
            x=tf.TensorSpec([None, 80], tf.int64),
            y=tf.TensorSpec([None, 80], tf.int64))
        return simple_fedavg_tf.KerasModelWrapper(keras_model, element_spec, loss)

    def tff_cifar_model_fn():
        """Constructs a fully initialized model for use in federated averaging."""
        keras_model = create_resnet18(CROP_SHAPE, NUM_CLASSES)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        element_spec = collections.OrderedDict(
            x=tf.TensorSpec([None, 32, 32, 3], tf.float32),
            y=tf.TensorSpec([None], tf.int64))
        return simple_fedavg_tf.KerasModelWrapper(keras_model, element_spec, loss)

    # def tff_shakespeare_model_fn():
    #     model_builder = functools.partial(
    #         models.create_recurrent_model, vocab_size=VOCAB_SIZE, sequence_length=dataset_shakespeare.SEQUENCE_LENGTH)
    #     loss_builder = functools.partial(
    #         tf.keras.losses.SparseCategoricalCrossentropy, from_logits=True)
    #
    #     def metrics_builder():
    #         """Returns a `list` of `tf.keras.metric.Metric` objects."""
    #         pad_token, _, _, _ = dataset_shakespeare.get_special_tokens()
    #
    #         return [
    #             keras_metrics.NumBatchesCounter(),
    #             keras_metrics.NumExamplesCounter(),
    #             keras_metrics.NumTokensCounter(masked_tokens=[pad_token]),
    #             keras_metrics.MaskedCategoricalAccuracy(masked_tokens=[pad_token]),
    #         ]
    #
    #     element_spec = collections.OrderedDict(
    #         x=tf.TensorSpec([None, 80], tf.int64),
    #         y=tf.TensorSpec([None, 80], tf.int64))
    #
    #     return tff.learning.from_keras_model(
    #         keras_model=model_builder(),
    #         input_spec=element_spec,
    #         loss=loss_builder(),
    #         metrics=metrics_builder())

    tff_model_fn = None
    if FLAGS.dataset == DATASET_FEMNIST:
        tff_model_fn = tff_emnist_model_fn
    elif FLAGS.dataset == DATASET_SHAKESPEARE:
        tff_model_fn = tff_shakespeare_model_fn
    elif FLAGS.dataset == DATASET_CIFAR10:
        tff_model_fn = tff_cifar_model_fn
    return tff_model_fn


def get_metric():
    metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    if FLAGS.dataset == DATASET_SHAKESPEARE:
        """Returns a `list` of `tf.keras.metric.Metric` objects."""
        pad_token, _, _, _ = dataset_shakespeare.get_special_tokens()

        metric = keras_metrics.MaskedCategoricalAccuracy(masked_tokens=[pad_token])
    return metric


def main(argv):
    if FLAGS.gpu == 0:
        print("Use CPU")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    start_time = time.time()

    np.random.seed(FLAGS.seed)
    tf.random.set_seed(FLAGS.seed)

    print("args", argv)
    print("FLAGS", FLAGS.flags_into_string())
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # If GPU is provided, TFF will by default use the first GPU like TF. The
    # following lines will configure TFF to use multi-GPUs and distribute client
    # computation on the GPUs. Note that we put server computatoin on CPU to avoid
    # potential out of memory issue when a large number of clients is sampled per
    # round. The client devices below can be an empty list when no GPU could be
    # detected by TF.
    server_device = tf.config.list_logical_devices('CPU')[0]

    if FLAGS.gpu == 0:
        client_devices = tf.config.list_logical_devices('CPU')
        print("client_devices", client_devices)
    elif FLAGS.gpu == 1:
        client_devices = [tf.config.list_logical_devices('GPU')[0]]
    else:
        client_devices = tf.config.list_logical_devices('GPU')

    tff.backends.native.set_local_execution_context(
        server_tf_device=server_device, client_tf_devices=client_devices)

    train_data, test_data = get_dataset()

    tff_model_fn = get_model_fn()

    iterative_process = simple_fedavg_tff.build_federated_averaging_process(
        tff_model_fn, server_optimizer_fn, client_optimizer_fn)
    server_state = iterative_process.initialize()

    model = tff_model_fn()

    metric = get_metric()

    cumulative_accuracies = []
    cumulative_training_times = []

    for round_num in range(FLAGS.total_rounds):
        np.random.seed(round_num)
        sampled_clients = np.random.choice(
            train_data.client_ids,
            size=FLAGS.train_clients_per_round,
            replace=False)

        sampled_train_data = [
            train_data.create_tf_dataset_for_client(client)
            for client in sampled_clients
        ]

        sampled_test_data = test_data
        if not FLAGS.test_in_server:
            clients_to_test = sampled_clients
            if FLAGS.test_all:
                clients_to_test = test_data.client_ids

            sampled_test_data = [
                test_data.create_tf_dataset_for_client(client)
                for client in clients_to_test
            ]

        server_state, train_metrics = iterative_process.next(server_state, sampled_train_data)
        print(f'Round {round_num} training loss: {train_metrics}')
        if round_num % FLAGS.rounds_per_eval == 0:
            model.from_weights(server_state.model_weights)
            weights = []
            accuracies = []
            if FLAGS.test_in_server:
                accuracy, _ = simple_fedavg_tf.keras_evaluate(model.keras_model, sampled_test_data, metric)
                accuracy = accuracy.numpy()
            else:
                for data in sampled_test_data:
                    accuracy, data_amount = simple_fedavg_tf.keras_evaluate(model.keras_model, data, metric)
                    accuracies.append(accuracy)
                    weights.append(data_amount)
                accuracy = np.average(accuracies, weights=weights)
            # accuracy = simple_fedavg_tf.keras_evaluate(model.keras_model, test_data, metric)
            print(f'Round {round_num} validation accuracy: {accuracy * 100.0}')
            cumulative_accuracies.append(accuracy * 100.0)
            cumulative_training_times.append(time.time() - start_time)
    print("Cumulative accuracies:", cumulative_accuracies)
    print("Cumulative training times:", cumulative_training_times)
    print("Total training time:", time.time() - start_time)


if __name__ == '__main__':
    app.run(main)
