import collections
import json
import os
from collections import defaultdict

import tensorflow as tf
import pickle

from tensorflow_federated.python.simulation import client_data


def load_dict(filename):
    with open(filename, 'rb') as f:
        dic = pickle.load(f)
    return dic


def load_data(data_dir, dataset):
    if data_dir == "":
        train_data_dir = os.path.join('..', 'data', dataset, 'data', 'train')
        test_data_dir = os.path.join('..', 'data', dataset, 'data', 'test')
    else:
        train_data_dir = os.path.join(data_dir, "train")
        test_data_dir = os.path.join(data_dir, "test")

    users, train_data, test_data = read_data(train_data_dir, test_data_dir, dataset)
    return CustomizedDataset(train_data, users, dataset), CustomizedDataset(test_data, users, dataset)


def read_data(train_data_dir, test_data_dir, dataset):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''

    if dataset == "cifar10":
        train_data = load_dict(train_data_dir)
        test_data = load_dict(test_data_dir)
        return list(train_data.keys()), train_data, test_data
    train_clients, _, train_data = read_dir(train_data_dir)
    test_clients, _, test_data = read_dir(test_data_dir)

    return train_clients, train_data, test_data


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


class CustomizedDataset(client_data.ClientData):
    def __init__(self, data, clients, dataset="femnist"):
        self._data = data
        self._dataset = dataset
        self._client_ids = clients
        self._element_type_structure = tf.TensorSpec(dtype=tf.string, shape=())

    def _create_dataset(self, client_id):
        """Creates a `tf.data.Dataset` for a client in a TF-serializable manner."""

        dic = collections.OrderedDict()
        for name, ds in self._data[client_id].items():
            if self._dataset == "femnist":
                if name == 'x':
                    dic['pixels'] = tf.reshape(ds, [-1, 28, 28])
                if name == 'y':
                    dic['label'] = ds[:]
            elif self._dataset == "shakespeare":
                if name == 'x':
                    dic['snippets'] = ds[:]
            elif self._dataset == "cifar10":
                if name == "x":
                    dic["image"] = ds[:]
                if name == "y":
                    dic["label"] = ds[:]
            else:
                dic[name] = ds[:]

        return tf.data.Dataset.from_tensor_slices(dic)

    @property
    def serializable_dataset_fn(self):
        return self._create_dataset

    @property
    def client_ids(self):
        return self._client_ids

    def create_tf_dataset_from_all_clients(self, seed=None):
        dic = collections.OrderedDict()
        for name, ds in self._data.items():
            if self._dataset == "cifar10":
                if name == "x":
                    dic["image"] = ds[:]
                if name == "y":
                    dic["label"] = tf.cast(ds[:], tf.int64)
        return tf.data.Dataset.from_tensor_slices(dic)
        # example_dataset = nested_dataset.flat_map(lambda x: x)
        # return example_dataset


    def create_tf_dataset_for_client(self, client_id: str):
        """Creates a new `tf.data.Dataset` containing the client training examples.

        This function will create a dataset for a given client if `client_id` is
        contained in the `client_ids` property of the `SQLClientData`. Unlike
        `self.serializable_dataset_fn`, this method is not serializable.

        Args:
          client_id: The string identifier for the desired client.

        Returns:
          A `tf.data.Dataset` object.
        """
        if client_id not in self.client_ids:
            raise ValueError(
                "ID [{i}] is not a client in this ClientData. See "
                "property `client_ids` for the list of valid ids.".format(
                    i=client_id))
        return self._create_dataset(client_id)

    @property
    def element_type_structure(self):
        return self._element_type_structure

    @property
    def dataset_computation(self):
        raise NotImplementedError("b/162106885")
