# Copyright 2021, The TensorFlow Federated Authors.
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

import collections

from absl import flags
import tensorflow as tf

from tensorflow_federated.python.simulation.datasets import celeba

FLAGS = flags.FLAGS
_EXPECTED_TYPE = collections.OrderedDict(
    sorted([(celeba.IMAGE_NAME,
             tf.TensorSpec(shape=(84, 84, 3), dtype=tf.int64))] +
           [(field_name, tf.TensorSpec(shape=(), dtype=tf.bool))
            for field_name in celeba.ATTRIBUTE_NAMES]))


class CelebATest(tf.test.TestCase):

  def test_load_from_gcs(self):
    self.skipTest(
        "CI infrastructure doesn't support downloading from GCS. Remove "
        'skipTest to run test locally.')
    celeba_test = celeba.load_data(FLAGS.test_tmpdir)[1]
    self.assertLen(celeba_test.client_ids, 935)
    self.assertIsInstance(celeba_test.element_type_structure,
                          collections.OrderedDict)
    self.assertEqual(_EXPECTED_TYPE, celeba_test.element_type_structure)

    # Check that clients have at least 5 examples. To do this check for every
    # client takes way too long and makes the unit test run time painful, so
    # just check the first ten clients.
    for client_id in celeba_test.client_ids[:10]:
      client_data = self.evaluate(
          list(celeba_test.create_tf_dataset_for_client(client_id)))
      self.assertGreaterEqual(len(client_data), 5)


if __name__ == '__main__':
  tf.test.main()
