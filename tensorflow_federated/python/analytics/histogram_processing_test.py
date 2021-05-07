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

from tensorflow_federated.python.analytics import histogram_processing
from tensorflow_federated.python.analytics import histogram_testcase


class HeavyHittersUtilsTest(histogram_testcase.HistogramTest):

  def test_threshold_histogram(self):
    histogram_keys = tf.constant([b'a', b'b', b'c', b'd', b'e'],
                                 dtype=tf.string)
    histogram_values = tf.constant([1.0, 10.3, 6.0, 4.5, 2.1], dtype=tf.float32)
    threshold = tf.constant(4.5, dtype=tf.float32)
    histogram_keys_thresholded_tf, histogram_values_thresholded_tf = histogram_processing.threshold_histogram(
        histogram_keys, histogram_values, threshold)
    expected_histogram = {b'b': 10.3, b'c': 6.0, b'd': 4.5}
    self.assertHistogramsAllClose(histogram_keys_thresholded_tf,
                                  histogram_values_thresholded_tf,
                                  expected_histogram.keys(),
                                  expected_histogram.values())

  def test_threshold_histogram_non_tensor_input(self):
    histogram_keys = [b'a', b'b', b'c', b'd', b'e']
    histogram_values = [1.0, 10.3, 6.0, 4.5, 2.1]
    threshold = 4.5
    histogram_keys_thresholded_tf, histogram_values_thresholded_tf = histogram_processing.threshold_histogram(
        histogram_keys, histogram_values, threshold)
    expected_histogram = {b'b': 10.3, b'c': 6.0, b'd': 4.5}
    self.assertHistogramsAllClose(histogram_keys_thresholded_tf,
                                  histogram_values_thresholded_tf,
                                  expected_histogram.keys(),
                                  expected_histogram.values())

  def test_threshold_histogram_inf_histogram_value(self):
    histogram_keys = [b'a', b'b', b'c', b'd', b'e']
    histogram_values = [1.0, float('inf'), 6.0, 4.5, 2.1]
    threshold = 4.5
    histogram_keys_thresholded_tf, histogram_values_thresholded_tf = histogram_processing.threshold_histogram(
        histogram_keys, histogram_values, threshold)
    expected_histogram = {b'b': float('inf'), b'c': 6.0, b'd': 4.5}
    self.assertHistogramsAllClose(histogram_keys_thresholded_tf,
                                  histogram_values_thresholded_tf,
                                  expected_histogram.keys(),
                                  expected_histogram.values())

  def test_threshold_histogram_inf_threshold(self):
    histogram_keys = [b'a', b'b', b'c', b'd', b'e']
    histogram_values = [1.0, 10.3, 6.0, 4.5, 2.1]
    threshold = float('inf')
    histogram_keys_thresholded_tf, histogram_values_thresholded_tf = histogram_processing.threshold_histogram(
        histogram_keys, histogram_values, threshold)
    expected_histogram = {}
    self.assertHistogramsAllClose(histogram_keys_thresholded_tf,
                                  histogram_values_thresholded_tf,
                                  expected_histogram.keys(),
                                  expected_histogram.values())

  def test_threshold_histogram_float64(self):
    histogram_keys = tf.constant([b'a', b'b', b'c', b'd', b'e'],
                                 dtype=tf.string)
    histogram_values = tf.constant([1.0, 10.3, 6.0, 4.5, 2.1], dtype=tf.float64)
    threshold = tf.constant(4.5, dtype=tf.float64)
    histogram_keys_thresholded_tf, histogram_values_thresholded_tf = histogram_processing.threshold_histogram(
        histogram_keys, histogram_values, threshold)
    expected_histogram = {b'b': 10.3, b'c': 6.0, b'd': 4.5}
    self.assertHistogramsAllClose(histogram_keys_thresholded_tf,
                                  histogram_values_thresholded_tf,
                                  expected_histogram.keys(),
                                  expected_histogram.values())

  def test_threshold_histogram_mix_input_type(self):
    histogram_keys = [b'a', b'b', b'c', b'd', b'e']
    histogram_values = tf.constant([1.0, 10.3, 6.0, 4.5, 2.1], dtype=tf.float32)
    threshold = 4.5
    histogram_keys_thresholded_tf, histogram_values_thresholded_tf = histogram_processing.threshold_histogram(
        histogram_keys, histogram_values, threshold)
    expected_histogram = {b'b': 10.3, b'c': 6.0, b'd': 4.5}
    self.assertHistogramsAllClose(histogram_keys_thresholded_tf,
                                  histogram_values_thresholded_tf,
                                  expected_histogram.keys(),
                                  expected_histogram.values())

  def test_threshold_histogram_duplicate_keys(self):
    histogram_keys = [b'a', b'a', b'b', b'c', b'd', b'e']
    histogram_values = tf.constant([1.0, 5.0, 10.3, 6.0, 4.5, 2.1],
                                   dtype=tf.float32)
    threshold = 4.5
    histogram_keys_thresholded_tf, histogram_values_thresholded_tf = histogram_processing.threshold_histogram(
        histogram_keys, histogram_values, threshold)

    expected_histogram_keys = [b'a', b'b', b'c', b'd']
    expected_histogram_values = [5.0, 10.3, 6.0, 4.5]

    # The (key, value) pairs are correctly thresholded when there are duplicate
    # keys, but values for the same key are not automatically summed up.
    self.assertAllEqual(histogram_keys_thresholded_tf, expected_histogram_keys)
    self.assertAllClose(histogram_values_thresholded_tf,
                        expected_histogram_values)

  def test_threshold_histogram_empty_inputs(self):
    histogram_keys = tf.constant([], dtype=tf.string)
    histogram_values = tf.constant([], dtype=tf.float32)
    threshold = tf.constant(0.0, dtype=tf.float32)
    histogram_keys_thresholded_tf, histogram_values_thresholded_tf = histogram_processing.threshold_histogram(
        histogram_keys, histogram_values, threshold)
    expected_histogram = {}
    self.assertHistogramsAllClose(histogram_keys_thresholded_tf,
                                  histogram_values_thresholded_tf,
                                  expected_histogram.keys(),
                                  expected_histogram.values())

  def test_threshold_histogram_len_mismatch(self):
    histogram_keys = tf.constant([b'a', b'b', b'c', b'd', b'e'],
                                 dtype=tf.string)
    histogram_values = tf.constant([1.0, 10.3, 6.0, 4.5, 2.1, 3, 6],
                                   dtype=tf.float32)
    threshold = tf.constant(4.5, dtype=tf.float32)
    with self.assertRaises(ValueError):
      _, _ = histogram_processing.threshold_histogram(histogram_keys,
                                                      histogram_values,
                                                      threshold)

  def test_threshold_histogram_dimension_mismatch(self):
    histogram_keys = tf.constant([[b'a', b'b', b'c', b'd', b'e']],
                                 dtype=tf.string)
    histogram_values = tf.constant([1.0, 10.3, 6.0, 4.5, 2.1, 3],
                                   dtype=tf.float32)
    threshold = tf.constant(4.5, dtype=tf.float32)
    with self.assertRaises(ValueError):
      _, _ = histogram_processing.threshold_histogram(histogram_keys,
                                                      histogram_values,
                                                      threshold)


if __name__ == '__main__':
  tf.test.main()
