from trax.layers import*


Latent_METRICS  = {
    'next_state_loss': Serial(Select([0,1,9]),
                              WeightedCategoryCrossEntropy()),
    'recon_state_loss': Serial(Select([2,3,10]),
                               WeightedCategoryCrossEntropy()),
    'recon_action_loss': Serial(Select([4,5,11]),
                                WeightedCategoryCrossEntropy()),
    'next_state_accuracy': Serial(Select([0,1,9]),
                                  Accuracy()),
    'recon_state_accuracy': Serial(Select([2,3,10]),
                                   Accuracy()),
    'recon_action_accuracy': Serial(Select([4,5,11]),
                                    Accuracy()),
    'next_state_sequence_accuracy': Serial(Select([0,1,9]),
                                           SequenceAccuracy()),
    'recon_state_sequence_accuracy': Serial(Select([2,3,10]),
                                            SequenceAccuracy()),
    'recon_action_sequence_accuracy': Serial(Select([4,5,11]),
                                             SequenceAccuracy()),
    # 'neg_log_perplexity': Serial(WeightedCategoryCrossEntropy(),
    #                                 Negate()),
    # 'weights_per_batch_per_core': Serial(Drop(), Drop(), Sum()),
}


@gin.configurable
def latent_fn():
  return Latent_METRICS


def _category_cross_entropy(model_output, targets):  # pylint: disable=invalid-name
  target_distributions = core.one_hot(targets, model_output.shape[-1])
  model_log_distributions = core.log_softmax(model_output)
  return - jnp.sum(target_distributions * model_log_distributions, axis=-1)


def LatentLossFunction():
  """
  Computes loss for the latent world model:
  1. transition loss                     (st_1)
  2. reconstruction of current goal        (st)
  3. reconstruction of current action      (at)
  4. predicting whether the action is valid for the current goal

  """
  def f(pred_s1, tok_s1, pred_s, tok_s, pred_a, tok_a, pred_v, v, r, w_st1, w_st, w_at):
    def _cat_loss(out, tgt, weights):
      cross_entropies =  _category_cross_entropy(out, tgt)
      return jnp.sum(cross_entropies * weights) / jnp.sum(weights)
    cross_entropies_st1 = _cat_loss(pred_s1, tok_s1, w_st1)
    cross_entropies_st = _cat_loss(pred_s, tok_s, w_st)
    cross_entropies_at = _cat_loss(pred_a, tok_a, w_at)
    loss = cross_entropies_st1 + cross_entropies_at + cross_entropies_st
    return loss

  return base.Fn('LatentLossFunction', f)