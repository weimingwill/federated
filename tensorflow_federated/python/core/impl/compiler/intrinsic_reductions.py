# Copyright 2018, The TensorFlow Federated Authors.
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
"""A library of transformations that can be applied to a computation."""

import collections
from typing import Callable, Dict

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_analysis


def _reduce_intrinsic(
    comp, uri, body_fn: Callable[[building_blocks.ComputationBuildingBlock],
                                 building_blocks.ComputationBuildingBlock]):
  """Replaces all the intrinsics with the given `uri` with a callable."""
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(uri, str)

  def _should_transform(comp):
    return comp.is_intrinsic() and comp.uri == uri

  def _transform(comp):
    if not _should_transform(comp):
      return comp, False
    arg_name = next(building_block_factory.unique_name_generator(comp))
    comp_arg = building_blocks.Reference(arg_name,
                                         comp.type_signature.parameter)
    intrinsic_body = body_fn(comp_arg)
    intrinsic_reduced = building_blocks.Lambda(comp_arg.name,
                                               comp_arg.type_signature,
                                               intrinsic_body)
    return intrinsic_reduced, True

  return transformation_utils.transform_postorder(comp, _transform)


def _apply_generic_op(op, arg):
  if not (arg.type_signature.is_federated() or
          type_analysis.is_structure_of_tensors(arg.type_signature)):
    # If there are federated elements nested in a struct, we need to zip these
    # together before passing to binary operator constructor.
    arg = building_block_factory.create_federated_zip(arg)
  return building_block_factory.apply_binary_operator_with_upcast(arg, op)


def get_intrinsic_reductions(
) -> Dict[str, Callable[[building_blocks.ComputationBuildingBlock],
                        building_blocks.ComputationBuildingBlock]]:
  """Returns map from intrinsic to reducing function.

  The returned dictionary is a `collections.OrderedDict` which maps intrinsic
  URIs to functions from building-block intrinsic arguments to an implementation
  of the intrinsic call which has been reduced to a smaller, more fundamental
  set of intrinsics.

  Bodies generated by later dictionary entries will not contain references
  to intrinsics whose entries appear earlier in the dictionary. This property
  is useful for simple reduction of an entire computation by iterating through
  the map of intrinsics, substituting calls to each.
  """

  # TODO(b/122728050): Implement reductions that follow roughly the following
  # breakdown in order to minimize the number of intrinsics that backends need
  # to support and maximize opportunities for merging processing logic to keep
  # the number of communication phases as small as it is practical. Perform
  # these reductions before FEDERATED_SUM (more reductions documented below).
  #
  # - FEDERATED_AGGREGATE(x, zero, accu, merge, report) :=
  #     GENERIC_MAP(
  #       GENERIC_REDUCE(
  #         GENERIC_PARTIAL_REDUCE(x, zero, accu, INTERMEDIATE_AGGREGATORS),
  #         zero, merge, SERVER),
  #       report)
  #
  # - FEDERATED_APPLY(f, x) := GENERIC_APPLY(f, x)
  #
  # - FEDERATED_BROADCAST(x) := GENERIC_BROADCAST(x, CLIENTS)
  #
  # - FEDERATED_COLLECT(x) := GENERIC_COLLECT(x, SERVER)
  #
  # - FEDERATED_MAP(f, x) := GENERIC_MAP(f, x)
  #
  # - FEDERATED_VALUE_AT_CLIENTS(x) := GENERIC_PLACE(x, CLIENTS)
  #
  # - FEDERATED_VALUE_AT_SERVER(x) := GENERIC_PLACE(x, SERVER)

  def generic_divide(arg):
    """Divides two arguments when possible."""
    py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
    return _apply_generic_op(tf.divide, arg)

  def generic_multiply(arg):
    """Multiplies two arguments when possible."""
    py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
    return _apply_generic_op(tf.multiply, arg)

  def generic_plus(arg):
    """Adds two arguments when possible."""
    py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
    return _apply_generic_op(tf.add, arg)

  def federated_weighted_mean(arg):
    py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
    w = building_blocks.Selection(arg, index=1)
    multiplied = generic_multiply(arg)
    zip_arg = building_blocks.Struct([(None, multiplied), (None, w)])
    summed = federated_sum(building_block_factory.create_federated_zip(zip_arg))
    return generic_divide(summed)

  def federated_mean(arg):
    py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
    one = building_block_factory.create_generic_constant(arg.type_signature, 1)
    mean_arg = building_blocks.Struct([(None, arg), (None, one)])
    return federated_weighted_mean(mean_arg)

  def federated_sum(x):
    py_typecheck.check_type(x, building_blocks.ComputationBuildingBlock)
    operand_type = x.type_signature.member
    zero = building_block_factory.create_generic_constant(operand_type, 0)
    plus_op = building_block_factory.create_tensorflow_binary_operator_with_upcast(
        computation_types.StructType([operand_type, operand_type]), tf.add)
    identity = building_block_factory.create_compiled_identity(operand_type)
    return building_block_factory.create_federated_aggregate(
        x, zero, plus_op, plus_op, identity)

  # - FEDERATED_ZIP(x, y) := GENERIC_ZIP(x, y)
  #
  # - GENERIC_AVERAGE(x: {T}@p, q: placement) :=
  #     GENERIC_WEIGHTED_AVERAGE(x, GENERIC_ONE, q)
  #
  # - GENERIC_WEIGHTED_AVERAGE(x: {T}@p, w: {U}@p, q: placement) :=
  #     GENERIC_MAP(GENERIC_DIVIDE, GENERIC_SUM(
  #       GENERIC_MAP(GENERIC_MULTIPLY, GENERIC_ZIP(x, w)), p))
  #
  #     Note: The above formula does not account for type casting issues that
  #     arise due to the interplay betwen the types of values and weights and
  #     how they relate to types of products and ratios, and either the formula
  #     or the type signatures may need to be tweaked.
  #
  # - GENERIC_SUM(x: {T}@p, q: placement) :=
  #     GENERIC_REDUCE(x, GENERIC_ZERO, GENERIC_PLUS, q)
  #
  # - GENERIC_PARTIAL_SUM(x: {T}@p, q: placement) :=
  #     GENERIC_PARTIAL_REDUCE(x, GENERIC_ZERO, GENERIC_PLUS, q)
  #
  # - GENERIC_AGGREGATE(
  #     x: {T}@p, zero: U, accu: <U,T>->U, merge: <U,U>=>U, report: U->R,
  #     q: placement) :=
  #     GENERIC_MAP(report, GENERIC_REDUCE(x, zero, accu, q))
  #
  # - GENERIC_REDUCE(x: {T}@p, zero: U, op: <U,T>->U, q: placement) :=
  #     GENERIC_MAP((a -> SEQUENCE_REDUCE(a, zero, op)), GENERIC_COLLECT(x, q))
  #
  # - GENERIC_PARTIAL_REDUCE(x: {T}@p, zero: U, op: <U,T>->U, q: placement) :=
  #     GENERIC_MAP(
  #       (a -> SEQUENCE_REDUCE(a, zero, op)), GENERIC_PARTIAL_COLLECT(x, q))
  #
  # - SEQUENCE_SUM(x: T*) :=
  #     SEQUENCE_REDUCE(x, GENERIC_ZERO, GENERIC_PLUS)
  #
  # After performing the full set of reductions, we should only see instances
  # of the following intrinsics in the result, all of which are currently
  # considered non-reducible, and intrinsics such as GENERIC_PLUS should apply
  # only to non-federated, non-sequence types (with the appropriate calls to
  # GENERIC_MAP or SEQUENCE_MAP injected).
  #
  # - GENERIC_APPLY
  # - GENERIC_BROADCAST
  # - GENERIC_COLLECT
  # - GENERIC_DIVIDE
  # - GENERIC_MAP
  # - GENERIC_MULTIPLY
  # - GENERIC_ONE
  # - GENERIC_ONLY
  # - GENERIC_PARTIAL_COLLECT
  # - GENERIC_PLACE
  # - GENERIC_PLUS
  # - GENERIC_ZERO
  # - GENERIC_ZIP
  # - SEQUENCE_MAP
  # - SEQUENCE_REDUCE

  return collections.OrderedDict([
      (intrinsic_defs.FEDERATED_MEAN.uri, federated_mean),
      (intrinsic_defs.FEDERATED_WEIGHTED_MEAN.uri, federated_weighted_mean),
      (intrinsic_defs.FEDERATED_SUM.uri, federated_sum),
      (intrinsic_defs.GENERIC_DIVIDE.uri, generic_divide),
      (intrinsic_defs.GENERIC_MULTIPLY.uri, generic_multiply),
      (intrinsic_defs.GENERIC_PLUS.uri, generic_plus),
  ])


def replace_intrinsics_with_bodies(comp):
  """Iterates over all intrinsic bodies, inlining the intrinsics in `comp`.

  This function operates on the AST level; meaning, it takes in a
  `building_blocks.ComputationBuildingBlock` as an argument and
  returns one as well. `replace_intrinsics_with_bodies` is intended to be the
  standard reduction function, which will reduce all currently implemented
  intrinsics to their bodies.

  Notice that the success of this function depends on the contract of
  `intrinsic_bodies.get_intrinsic_bodies`, that the dict returned by that
  function is ordered from more complex intrinsic to less complex intrinsics.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock` in which we
      wish to replace all intrinsics with their bodies.

  Returns:
    Instance of `building_blocks.ComputationBuildingBlock` with all
    the intrinsics from `intrinsic_bodies.py` inlined with their bodies, along
    with a Boolean indicating whether there was any inlining in fact done.

  Raises:
    TypeError: If the types don't match.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  bodies = get_intrinsic_reductions()
  transformed = False
  for uri, body in bodies.items():
    comp, uri_found = _reduce_intrinsic(comp, uri, body)
    transformed = transformed or uri_found
  return comp, transformed
