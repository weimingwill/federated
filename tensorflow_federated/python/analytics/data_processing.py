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
"""A set of utility functions for data processing."""

import math

import tensorflow as tf


@tf.function
def get_all_elements(dataset, batched=False):
  """Gets all the elements from the input dataset.

  Args:
    dataset: A `tf.data.Dataset` containing a list of elements.
    batched: Whether client dataset is batched by `tf.data.Dataset.batch()` or
      not.

  Returns:
    A tensor of a list of all the elements in the input dataset.
  """

  element_type = dataset.element_spec.dtype
  initial_list = tf.constant([], dtype=element_type)

  def add_element(element_list, input_item):
    if batched:
      new_element_list = input_item
    else:
      new_element_list = tf.expand_dims(input_item, axis=0)
    element_list = tf.concat([element_list, new_element_list], axis=0)
    return element_list

  all_element_list = dataset.reduce(
      initial_state=initial_list, reduce_func=add_element)

  return all_element_list


@tf.function
def get_capped_elements(dataset,
                        max_user_contribution,
                        batched=False,
                        batch_size=1):
  """Gets the first max_user_contribution words from the input list.

  Note that if the dataset is batched, either none of the elements in one batch
  is added to the result, or all the elements are added. This means the length
  of the returned list of elements could be less than `max_user_contribution`
  even when `dataset` is capped.

  Args:
    dataset: A `tf.data.Dataset` containing a list of elements.
    max_user_contribution: The maximum number of elements to keep.
    batched: Whether client dataset is batched by `tf.data.Dataset.batch()` or
      not.
    batch_size: The number of items in each batch. This value is ignored when
      `batched` is `False`.

  Returns:
    A tensor of a list of strings.
    If the total number of words is less than or equal to
    max_user_contribution, returns all the words in the dataset.
  """
  if batched:
    capped_size = math.floor(max_user_contribution / batch_size)
  else:
    capped_size = max_user_contribution

  capped_dataset = dataset.take(capped_size)
  return get_all_elements(capped_dataset, batched=batched)


@tf.function
def get_unique_elements(dataset, batched=False):
  """Gets the unique words from the input list.

  Args:
    dataset: A `tf.data.Dataset` containing a list of elements.
    batched: Whether dataset is batched by `tf.data.Dataset.batch()` or not.

  Returns:
    A tensor of a list of strings.
  """
  element_type = dataset.element_spec.dtype
  initial_list = tf.constant([], dtype=element_type)

  def add_unique_element(element_list, new_element):
    # This method is more memory efficient than creating a giant tensor and
    # use `tf.unique` since it only puts one element in memory each time. A
    # possible alternative way is to use `tf.data.experimental.unique`.
    mask = tf.equal(element_list, new_element)
    # If the element doesn't match any we've already seen, add it to the list.
    if not tf.reduce_any(mask):
      element_list = tf.concat(
          [element_list, tf.expand_dims(new_element, axis=0)], axis=0)
    return element_list

  def add_unique_element_batched(element_list, new_element_batch):
    element_list = tf.concat([element_list, new_element_batch], axis=0)
    element_list, _ = tf.unique(element_list)
    return element_list

  if batched:
    unique_element_list = dataset.reduce(
        initial_state=initial_list, reduce_func=add_unique_element_batched)
  else:
    unique_element_list = dataset.reduce(
        initial_state=initial_list, reduce_func=add_unique_element)

  return unique_element_list
