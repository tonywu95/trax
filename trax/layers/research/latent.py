import gin

import trax.layers as tl
from trax.fastmath import numpy as jnp

Latent_METRICS  = {
    'next_state_loss': tl.Serial(tl.Select([0,1,10]),
                              tl.WeightedCategoryCrossEntropy()),
    # 'recon_state_loss': tl.Serial(tl.Select([2,3,10]),
    #                                        tl.WeightedCategoryCrossEntropy()),
    # 'recon_action_loss': tl.Serial(tl.Select([4,5,11]),
    #                                         tl.WeightedCategoryCrossEntropy()),
    'next_state_accuracy': tl.Serial(tl.Select([0,1,10]),
                                              tl.Accuracy()),
    # 'recon_state_accuracy': tl.Serial(tl.Select([2,3,10]),
    #                                            tl.Accuracy()),
    # 'recon_action_accuracy': tl.Serial(tl.Select([4,5,11]),
    #                                             tl.Accuracy()),
    'next_state_sequence_accuracy': tl.Serial(tl.Select([0,1,10]),
                                                       tl.SequenceAccuracy()),
    # 'recon_state_sequence_accuracy': tl.Serial(tl.Select([2,3,10]),
    #                                                     tl.SequenceAccuracy()),
    # 'recon_action_sequence_accuracy': tl.Serial(tl.Select([4,5,11]),
    #                                                      tl.SequenceAccuracy()),
    # 'neg_log_perplexity': Serial(WeightedCategoryCrossEntropy(),
    #                                 Negate()),
    # 'weights_per_batch_per_core': Serial(Drop(), Drop(), Sum()),
}


@gin.configurable
def latent_fn():
  return Latent_METRICS


def _category_cross_entropy(model_output, targets):  # pylint: disable=invalid-name
  target_distributions = tl.core.one_hot(targets, model_output.shape[-1])
  model_log_distributions = tl.core.log_softmax(model_output)
  return - jnp.sum(target_distributions * model_log_distributions, axis=-1)


@gin.configurable
def LatentLossFunction():
  """
  Computes loss for the latent world model:
  1. transition loss                     (st_1)
  2. reconstruction of current goal        (st)
  3. reconstruction of current action      (at)
  4. predicting whether the action is valid for the current goal

  """
  # def f(pred_s1, tok_s1, pred_s, tok_s, pred_a, tok_a, pred_v, v, r, w_st1, w_st, w_at):
  def f(pred_s, tok_s, pred_s1, tok_s1, pred_a, tok_a, pred_v, v, r, w_st1, w_st, w_at):
    def _cat_loss(out, tgt, weights):
      cross_entropies =  _category_cross_entropy(out, tgt)
      return jnp.sum(cross_entropies * weights) / jnp.sum(weights)
    # cross_entropies_st1 = _cat_loss(pred_s1, tok_s1, w_st1)
    cross_entropies_st = _cat_loss(pred_s, tok_s, w_st)
    # cross_entropies_at = _cat_loss(pred_a, tok_a, w_at)
    loss = cross_entropies_st #+ cross_entropies_at + cross_entropies_st1
    return loss

  return tl.base.Fn('LatentLossFunction', f)

