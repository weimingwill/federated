import collections

import tensorflow as tf

from tensorflow_federated.python.examples.simple_fedavg.customized_dataset import load_data


def get_emnist_dataset(FLAGS):
    train_data, test_data = load_data(FLAGS.data_dir, "femnist")

    def element_fn(element):
        return collections.OrderedDict(
            x=tf.expand_dims(element['pixels'], -1), y=element['label'])

    def preprocess_train_dataset(dataset):
        # Use buffer_size same as the maximum client dataset size,
        # 418 for Federated EMNIST
        return dataset.map(element_fn).shuffle(buffer_size=418).repeat(
            count=FLAGS.client_epochs_per_round).batch(
            FLAGS.batch_size, drop_remainder=False)

    def preprocess_test_dataset(dataset):
        return dataset.map(element_fn).batch(
            FLAGS.test_batch_size, drop_remainder=False)

    train_data = train_data.preprocess(preprocess_train_dataset)
    test_data = test_data.preprocess(preprocess_test_dataset)
    # test_data = preprocess_test_dataset(
    #     test_data.create_tf_dataset_from_all_clients())
    return train_data, test_data
