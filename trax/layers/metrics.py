# coding=utf-8
# Copyright 2020 The Trax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Layers for computing loss functions and evaluation metrics.

A metric layer computes a scalar value from two or three ndarray inputs:

  - model outputs: Batch of predicted values (typically vectors).

  - targets: Batch of target values (e.g., categories or vectors).

  - weights: Float values that allow for uneven weighting of batch items,
    sequence positions, or vector components when computing an overall scalar
    value for the batch.

Most metric computations take into account the items that make up a batch. For
each item in a batch, a raw metric value is computed by comparing (item-wise)
the model output to the target value. These item-wise values are then combined
into a single scalar for the batch by a function such as sum, average, or
weighted-average. For example:

  - CategoryAccuracy: Treat model output as giving different strength/votes to
    the possible categories; measure the category prediction as correct (value
    1) if `argmax(output) == target_category`, else as incorrect (value 0). The
    accuracy for the batch is then the average of these 1's and 0's.

  - CategoryCrossEntropy: Treat model output and target values as the source of
    two probability distributions; measure the cross entropy of the model's
    predicted distribution relative to the (assumed true) target distribution.
    The scalar value for the batch is then the average of the item-wise
    cross-entropy values.
"""

from trax import shapes
from trax.fastmath import numpy as jnp
from trax.layers import base
from trax.layers import combinators as cb
from trax.layers import core


def CategoryAccuracy():
  """Returns a layer that computes category prediction accuracy.

  The layer takes two inputs:

    - A batch of activation vectors. The components in a given vector should
      be mappable to a probability distribution in the following loose sense:
      within a vector, a higher component value corresponds to a higher
      probability, such that argmax within a vector (`axis=-1`) picks the index
      (category) having the highest probablity.

    - A batch of target categories; each target is an integer in
      `{0, ..., N-1}`.

  The predicted category from each vector is the index of the highest-valued
  vector component. The layer returns the accuracy of these predictions
  averaged over the batch.
  """
  def f(model_output, targets):  # pylint: disable=invalid-name
    predictions = jnp.argmax(model_output, axis=-1)
    shapes.assert_same_shape(predictions, targets)
    n_total = predictions.size
    n_correct = jnp.sum(jnp.equal(predictions, targets))
    return n_correct / n_total

  return base.Fn('CategoryAccuracy', f)


def WeightedCategoryAccuracy():
  """Returns a layer that computes a weighted category prediction accuracy.

  The layer takes three inputs:

    - A batch of activation vectors. The components in a given vector should
      be mappable to a probability distribution in the following loose sense:
      within a vector, a higher component value corresponds to a higher
      probability, such that argmax within a vector (`axis=-1`) picks the index
      (category) having the highest probablity.

    - A batch of target categories; each target is an integer in
      `{0, ..., N-1}`.

    - A batch of weights, which matches or can be broadcast to match the shape
      of the target ndarray. This arg can give uneven weighting to different
      items in the batch (depending, for instance, on the item's target
      category).

  The predicted category from each vector is the index of the highest-valued
  vector component. The layer returns a weighted average accuracy of these
  predictions.
  """
  def f(model_output, targets, weights):  # pylint: disable=invalid-name
    predictions = jnp.argmax(model_output, axis=-1)
    shapes.assert_same_shape(predictions, targets)
    ones_and_zeros = jnp.equal(predictions, targets)
    return jnp.sum(ones_and_zeros * weights) / jnp.sum(weights)

  return base.Fn('WeightedCategoryAccuracy', f)


def CategoryCrossEntropy():
  """Returns a layer that computes cross entropy from activations and integers.

  The layer takes two inputs:

    - A batch of activation vectors. The components in a given vector should
      be pre-softmax activations (mappable to a probability distribution via
      softmax). For performance reasons, the softmax and cross entropy
      computations are combined inside the layer.

    - A batch of target categories; each target is an integer in
      `{0, ..., N-1}`, where `N` is the activation vector depth/dimensionality.

  To compute cross-entropy, the layer derives probability distributions from
  its inputs:

    - activation vectors: vector --> SoftMax(vector)

    - target categories: integer --> OneHot(integer)

  (The conversion of integer category targets to one-hot vectors amounts to
  assigning all the probability mass to the target category.) Cross-entropy
  per batch item is computed between the resulting distributions; notionally:

      cross_entropy(one_hot(targets), softmax(model_output))

  The layer returns the average of these cross-entropy values over all items in
  the batch.
  """
  def f(model_output, targets):  # pylint: disable=invalid-name
    cross_entropies = _category_cross_entropy(model_output, targets)
    return jnp.average(cross_entropies)

  return base.Fn('CategoryCrossEntropy', f)


def WeightedCategoryCrossEntropy():
  """Returns a layer like `CategoryCrossEntropy`, with weights as third input.

  The layer takes three inputs:

    - A batch of activation vectors. The components in a given vector should
      be pre-softmax activations (mappable to a probability distribution via
      softmax). For performance reasons, the softmax and cross entropy
      computations are combined inside the layer.

    - A batch of target categories; each target is an integer in
      `{0, ..., N-1}`, where `N` is the activation vector depth/dimensionality.

    - A batch of weights, which matches or can be broadcast to match the shape
      of the target ndarray. This arg can give uneven weighting to different
      items in the batch (depending, for instance, on the item's target
      category).

  To compute cross-entropy, the layer derives probability distributions from
  its inputs:

    - activation vectors: vector --> SoftMax(vector)

    - target categories: integer --> OneHot(integer)

  (The conversion of integer category targets to one-hot vectors amounts to
  assigning all the probability mass to the target category.) Cross-entropy
  per batch item is computed between the resulting distributions; notionally:

      cross_entropy(one_hot(targets), softmax(model_output))

  The layer returns the weighted average of these cross-entropy values over all
  items in the batch.
  """
  def f(model_output, targets, weights):  # pylint: disable=invalid-name
    cross_entropies = _category_cross_entropy(model_output, targets)
    return jnp.sum(cross_entropies * weights) / jnp.sum(weights)

  return base.Fn('WeightedCategoryCrossEntropy', f)


def Accuracy(classifier=core.ArgMax()):
  """Returns a layer that computes mean category prediction accuracy."""
  return cb.Serial(classifier,
                   _Accuracy(),
                   _WeightedMean(),
                   name='Accuracy',
                   sublayers_to_print=[])


def SequenceAccuracy(classifier=core.ArgMax()):
  """Returns a layer that computes mean sequence prediction accuracy."""
  return cb.Serial(classifier,
                   _Accuracy(),
                   _WeightedSequenceMean(),
                   name='SequenceAccuracy',
                   sublayers_to_print=[])


def CrossEntropyLoss():
  """Mean prediction-target cross entropy for multiclass classification."""
  return cb.Serial(_CrossEntropy(),
                   _WeightedMean(),
                   name='CrossEntropyLoss',
                   sublayers_to_print=[])


def CrossEntropyLossWithLogSoftmax():
  """Mean prediction-target cross entropy for multiclass classification."""
  return cb.Serial(core.LogSoftmax(), CrossEntropyLoss(),
                   name='CrossEntropyLoss')


def BinaryCrossEntropyLoss():
  """Mean prediction-target cross entropy for binary classification."""
  return cb.Serial(_BinaryCrossEntropy(),
                   _WeightedMean(),
                   name='BinaryCrossEntropyLoss',
                   sublayers_to_print=[])


def L2Loss():
  """Returns a layer that computes an L2-like loss for one batch."""
  def f(model_output, targets, weights):  # pylint: disable=invalid-name
    """Returns weighted sum-of-squared-errors for `model_output` vs. `targets`.

    Args:
      model_output: Output from one batch, typically a 2- or 3-d array of
          float-valued elements.
      targets: Tensor of same shape as `model_output` containing element-wise
          target values.
      weights: Tensor of same shape as `model_output` and `targets`, containing
          element-wise weight values.
    """
    shapes.assert_same_shape(model_output, targets)
    shapes.assert_same_shape(targets, weights)
    weighted_sse = weights * (model_output - targets)**2
    return jnp.sum(weighted_sse) / jnp.sum(weights)
  return base.Fn('L2Loss', f)


def SmoothL1Loss():
  """Returns a layer that computes total smooth L1 loss for one batch."""
  def smoothl1loss(model_output, targets, weights):  # pylint: disable=invalid-name
    r"""Returns weighted smooth L1 norm of `model_output - targets`.

    The smooth L1 loss, also known as the Huber loss, is defined as:
    .. math::
        z_i =
        \begin{cases}
        0.5 (x_i - y_i)^2, & \text{if } |x_i - y_i| < 1 \\
        |x_i - y_i| - 0.5, & \text{otherwise }
        \end{cases}

    Args:
      model_output: Output from one batch, treated as an unanalyzed tensor.
      targets: Tensor of same shape as `model_output` containing element-wise
          target values.
      weights: Tensor of same shape as `model_output` and `targets`, containing
          element-wise weight values.
    """
    shapes.assert_same_shape(model_output, targets)
    shapes.assert_same_shape(targets, weights)
    l1_dist = jnp.abs(model_output - targets)
    smooth_dist = jnp.where(l1_dist < 1,
                            0.5 * l1_dist**2,
                            l1_dist - 0.5)
    shapes.assert_same_shape(smooth_dist, weights)
    weighted_smooth_dist = weights * smooth_dist
    return jnp.sum(weighted_smooth_dist) / jnp.sum(weights)
  return base.Fn('SmoothL1Loss', smoothl1loss)


def WeightedSum():
  """Returns a layer that computes a weighted sum of the given values."""
  def f(values, weights):  # pylint: disable=invalid-name
    return jnp.sum(values * weights)
  return base.Fn('WeightedSum', f)


def _Accuracy():
  """Returns a layer that scores predicted versus target category."""
  def f(predicted_category, target_category):  # pylint: disable=invalid-name
    # TODO(pkozakowski): This assertion breaks some tests. Fix and uncomment.
    # shapes.assert_same_shape(predicted_category, target_category)
    return jnp.equal(predicted_category, target_category).astype(jnp.float32)
  return base.Fn('_Accuracy', f)


def _CrossEntropy():
  """Returns a layer that computes prediction-target cross entropies."""
  def f(model_output, target_category):  # pylint: disable=invalid-name
    # TODO(pkozakowski): This assertion breaks some tests. Fix and uncomment.
    # shapes.assert_shape_equals(target_category, model_output.shape[:-1])
    target_distribution = core.one_hot(target_category, model_output.shape[-1])
    return -1.0 * jnp.sum(model_output * target_distribution, axis=-1)
  return base.Fn('_CrossEntropy', f)


def _BinaryCrossEntropy():
  """Returns a layer that computes prediction-target cross entropies."""
  def f(model_output, target_category):  # pylint: disable=invalid-name
    shapes.assert_same_shape(model_output, target_category)
    batch_size = model_output.shape[0]
    j = jnp.dot(jnp.transpose(target_category), jnp.log(model_output))
    j += jnp.dot(jnp.transpose(1 - target_category), jnp.log(1 - model_output))
    j = -1.0/batch_size * jnp.squeeze(j)
    return j
  return base.Fn('_BinaryCrossEntropy', f)


def CrossEntropySum():
  """Sum of prediction-target cross entropies for multiclass classification."""
  return cb.Serial(_CrossEntropy(),
                   WeightedSum(),
                   name='CrossEntropySum',
                   sublayers_to_print=[])


def BinaryCrossEntropySum():
  """Sum of prediction-target cross entropies for binary classification."""
  return cb.Serial(_BinaryCrossEntropy(),
                   WeightedSum(),
                   name='BinaryCrossEntropySum',
                   sublayers_to_print=[])
# pylint: enable=no-value-for-parameter


def _WeightedMean():
  """Returns a layer that computes a weighted mean of the given values."""
  def f(values, weights):  # pylint: disable=invalid-name
    return jnp.sum(values * weights) / jnp.sum(weights)
  return base.Fn('_WeightedMean', f)


def _WeightedSequenceMean():
  """Returns a layer that computes a weighted sequence accuracy mean."""
  def f(values, weights):  # pylint: disable=invalid-name
    # This function assumes weights are 0 or 1.
    # Then compute 1: not-correct, 0: correct or masked
    not_correct = (1.0 - values) * weights
    axis_to_sum = list(range(1, len(not_correct.shape)))
    # Summing not-correct on all axes but batch. We're summing 0s and 1s,
    # so the sum is 0 if it's all 0 and >=1 in all other cases.
    not_correct_seq = jnp.sum(not_correct, axis=axis_to_sum)
    # Sequence is correct if not_correct_seq is 0, reverting here.
    correct_seq = 1.0 - jnp.minimum(1.0, not_correct_seq)
    return jnp.mean(correct_seq)  # Mean over batch.
  return base.Fn('_WeightedSequenceMean', f)


def _category_cross_entropy(model_output, targets):  # pylint: disable=invalid-name
  target_distributions = core.one_hot(targets, model_output.shape[-1])
  model_log_distributions = core.log_softmax(model_output)
  return - jnp.sum(target_distributions * model_log_distributions, axis=-1)
