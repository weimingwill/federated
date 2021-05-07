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

import tensorflow as tf

from tensorflow_federated.python.analytics import data_processing


class GetSelectedElementsTest(tf.test.TestCase):

  def test_all_elements(self):
    ds = tf.data.Dataset.from_tensor_slices(['a', 'b', 'a', 'b', 'c'])
    all_elements = data_processing.get_all_elements(ds)
    self.assertAllEqual(all_elements, [b'a', b'b', b'a', b'b', b'c'])

  def test_all_elements_batched(self):
    batch_size = 2
    ds = tf.data.Dataset.from_tensor_slices(['a', 'b', 'a', 'b',
                                             'c']).batch(batch_size)
    all_elements = data_processing.get_all_elements(ds, batched=True)
    self.assertAllEqual(all_elements, [b'a', b'b', b'a', b'b', b'c'])

  def test_capped_elements_empty_dataset(self):
    ds = tf.data.Dataset.from_tensor_slices([])
    capped_elements = data_processing.get_capped_elements(
        ds, max_user_contribution=10)
    self.assertEmpty(capped_elements)

  def test_capped_elements_under_max_contribution(self):
    ds = tf.data.Dataset.from_tensor_slices(['a', 'b', 'a', 'b', 'c'])
    capped_elements = data_processing.get_capped_elements(
        ds, max_user_contribution=10)
    self.assertAllEqual(capped_elements, [b'a', b'b', b'a', b'b', b'c'])

  def test_capped_elements_over_max_contribution(self):
    ds = tf.data.Dataset.from_tensor_slices(['a', 'b', 'a', 'c', 'b', 'c', 'c'])
    capped_elements = data_processing.get_capped_elements(
        ds, max_user_contribution=4)
    self.assertAllEqual(capped_elements, [b'a', b'b', b'a', b'c'])

  def test_capped_elements_batched(self):
    batch_size = 2
    ds = tf.data.Dataset.from_tensor_slices(['a', 'b', 'a', 'c', 'b', 'c',
                                             'c']).batch(batch_size)
    capped_elements = data_processing.get_capped_elements(
        ds, max_user_contribution=4, batched=True, batch_size=batch_size)
    self.assertAllEqual(capped_elements, [b'a', b'b', b'a', b'c'])

  def test_capped_elements_batched_2(self):
    batch_size = 3
    ds = tf.data.Dataset.from_tensor_slices(['a', 'b', 'a', 'c', 'b', 'c',
                                             'c']).batch(batch_size)
    capped_elements = data_processing.get_capped_elements(
        ds, max_user_contribution=4, batched=True, batch_size=batch_size)
    # Only returns the first 3 strings, because adding the second batch would
    # make the list size larger than `max_user_contribution`.
    self.assertAllEqual(capped_elements, [b'a', b'b', b'a'])

  def test_capped_elements_with_int(self):
    ds = tf.data.Dataset.from_tensor_slices([1, 3, 2, 2, 4, 6, 3])
    capped_elements = data_processing.get_capped_elements(
        ds, max_user_contribution=4)
    self.assertAllEqual(capped_elements, [1, 3, 2, 2])

  def test_unique_elements_empty_dataset(self):
    ds = tf.data.Dataset.from_tensor_slices([])
    unique_elements = data_processing.get_unique_elements(ds)
    self.assertEmpty(unique_elements)

  def test_unique_elements(self):
    ds = tf.data.Dataset.from_tensor_slices(['a', 'b', 'a', 'c', 'b', 'c', 'c'])
    unique_elements = data_processing.get_unique_elements(ds)
    self.assertAllEqual(unique_elements, [b'a', b'b', b'c'])

  def test_unique_elements_batched(self):
    batch_size = 3
    ds = tf.data.Dataset.from_tensor_slices(['a', 'b', 'a', 'c', 'b', 'c',
                                             'c']).batch(batch_size)
    unique_elements = data_processing.get_unique_elements(ds, batched=True)
    self.assertAllEqual(unique_elements, [b'a', b'b', b'c'])

  def test_unique_elements_with_int(self):
    ds = tf.data.Dataset.from_tensor_slices([1, 3, 2, 2, 4, 6, 3])
    unique_elements = data_processing.get_unique_elements(ds)
    self.assertAllEqual(unique_elements, [1, 3, 2, 4, 6])


if __name__ == '__main__':
  tf.test.main()
