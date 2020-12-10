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
"""Transformer models: encoder, decoder, language model, and encoder-decoder.

The "Transformer" name and network architecture were introduced in the paper
[Attention Is All You Need](https://arxiv.org/abs/1706.03762).
"""

from trax import layers as tl
from trax.layers.base import Fn
from trax.layers.assert_shape import assert_shape
from trax import fastmath
from trax.fastmath import numpy as jnp
from trax.layers import base
from trax.layers import core
import copy
from trax.models.research import configurable_transformer as ct

def LatentTransformer(input_vocab_size,
                 output_vocab_size=None,
                 d_model=512,
                 d_ff=2048,
                 n_encoder_layers=6,
                 n_decoder_layers=6,
                 n_heads=8,
                 dropout=0.1,
                 dropout_shared_axes=None,
                 max_len=2048,
                 mode='train',
                 ff_activation=tl.Relu,
                 ff_dropout=0.1,
                 ff_chunk_size=0,
                 ff_use_sru=0,
                 ff_sparsity=0,
                 ff_sparsity_type='1inN',
                 attention_chunk_size=0,
                 encoder_attention_type=tl.Attention,
                 n_encoder_attention_layers=1,
                 decoder_attention_type=tl.CausalAttention,
                 n_decoder_attention_layers=2,
                 axial_pos_shape=None,
                 d_axial_pos_embs=None):
  """Returns a Transformer model.

  This model expects an input pair: target, source.

  Args:
    input_vocab_size: int: vocab size of the source.
    output_vocab_size: int (optional): vocab size of the target. If None, the
      source and target are assumed to have the same vocab.
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    n_encoder_layers: int: number of encoder layers
    n_decoder_layers: int: number of decoder layers
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    dropout_shared_axes: axes on which to share dropout mask
    max_len: int: maximum symbol length for positional encoding
    mode: str: 'train' or 'eval'
    ff_activation: the non-linearity in feed-forward layer
    ff_dropout: Stochastic rate (probability) for dropping an activation value
      when applying dropout after the FF dense layer.
    ff_chunk_size: int; if > 0, chunk feed-forward into this-sized chunks
    ff_use_sru: int; if > 0, we use this many SRU layers instead of feed-forward
    ff_sparsity: int, if > 0 use sparse feed-forward block with this sparsity
    ff_sparsity_type: string, if ff_sparsity >0,
      use SparseFF if ff_sparsity_type=`'1inN'` and
      use BlockSparseFF if ff_sparsity_type=`'Block'`
    attention_chunk_size: int, if > 0 run attention chunked at this size
    encoder_attention_type: The attention layer to use for the encoder part.
    n_encoder_attention_layers: int, within each encoder block, how many
      attention layers to have.
    decoder_attention_type: The attention layer to use for the
      encoder-decoder attention.
    n_decoder_attention_layers: int, within each decoder block, how many
      attention layers to have.
    axial_pos_shape: tuple of ints: input shape to use for the axial position
      encoding. If unset, axial position encoding is disabled.
    d_axial_pos_embs: tuple of ints: depth of position embedding for each axis.
      Tuple length must match axial_pos_shape, and values must sum to d_model.

  Returns:
    A Transformer model as a layer that maps from a target, source pair to
    activations over a vocab set.
  """
  in_encoder, out_encoder, output_vocab_size = (
      ct.EmbeddingAndPositionalEncodings(
          input_vocab_size,
          d_model,
          mode,
          dropout,
          dropout_shared_axes,
          max_len,
          output_vocab_size=output_vocab_size,
          axial_pos_shape=axial_pos_shape,
          d_axial_pos_embs=d_axial_pos_embs)
  )

  # pylint: disable=g-complex-comprehension
  encoder_blocks = [
      _EncoderBlock(d_model, d_ff, n_heads,
                    dropout, dropout_shared_axes, mode, ff_activation)
      for i in range(n_encoder_layers)]
  # pylint: enable=g-complex-comprehension
  # padding = tl.Branch([], tl.PaddingMask)

  encoder = tl.Serial(
      # tl.Branch([], tl.PaddingMask()),  # tok_e mask_e tok_e tok_d tok_d
      in_encoder,
      encoder_blocks,
      tl.LayerNorm()
  )
  if mode == 'predict':
    encoder = tl.Cache(encoder)

  # pylint: disable=g-complex-comprehension
  decoder_blocks = [
      _DecoderBlock(d_model, d_ff, n_heads,
                    dropout, dropout_shared_axes, mode, ff_activation)
      for i in range(n_decoder_layers)]

  compress_seq = tl.Serial(
      # input:                            #   tok
      tl.Branch([], tl.PaddingMask()),    #   tok mask
      encoder,                            #   vec mask
      PickFirst(),                        # vec_f mask
      tl.Select([0], n_in=2)) # vec_f

  latent_transition = tl.Serial(
      tl.Parallel(
          [tl.Dense(d_model), tl.Relu()],
          [tl.Dense(d_model), tl.Relu()]
      ),
      tl.Add(),
      tl.Residual(
      tl.LayerNorm(),
      tl.Dense(d_model),
      tl.Relu(),
      tl.Dropout(rate=dropout, mode=mode),
      tl.Dense(d_model),
      ))

  pred_valid = tl.Serial(tl.Dense(2), Squeeze(1))

  embed_tgt = tl.Serial(
      # Input                             #  tok_d
      DropLast(mode=mode),                # stok_d
      out_encoder,                        # svec_d
  )

  decode_seq = tl.Serial(
      # Input:                                 #  vec_e  tok_d
      tl.Select([1,0, 1]),                     #  tok_d  vec_e tok_d
      tl.Parallel(embed_tgt, [],
                  DropFirst()),                # svec_d  vec_e tok_d'
      ConcatDeEntoEnDe(),                      # vec_ed tok_d'
      # Decoder blocks with causal attention
      decoder_blocks,                          # vec_ed tok_d'
      tl.LayerNorm(),                          # vec_ed tok_d'
      DropFirst(),                             #  vec_d tok_d'
      # Map to output vocab.
      tl.Dense(output_vocab_size),             # pred_d tok_d'
  )

  # compress_seq: n_in 1 n_out 1: add mask, encode, pick last hidden
  # latent_transition: n_in 2 n_out 1: s, a -> s_1
  # pred_valid: n_in 1 n_out 1: s_1 -> pred_v
  # decode_seq: n_in 2 n_out 2: copy target, shift right, decode, output


  return tl.Serial(
                                              #       0      1      2      3      4     5      6 7 8
      # Input:                                #   tok_s  tok_a tok_s1      r      v
      tl.Select([0, 1, 2, 0, 1, 3, 4]),       #   tok_s  tok_a tok_s1  tok_s  tok_a     r      v

      # Encode.
      tl.Parallel(compress_seq,
                  compress_seq),              #   vec_s  vec_a tok_s1  tok_s  tok_a     r      v
      tl.Branch(latent_transition,
                [], tl.Select([1], n_in=2)),  #  vec_s1  vec_s  vec_a tok_s1  tok_s tok_a      r v
      tl.Branch(pred_valid, []),              #  pred_v vec_s1  vec_s  vec_a tok_s1 tok_s  tok_a r v
      # Decode.
      tl.Select([1, 4, 2, 5, 3, 6, 0, 8, 7]), #  vec_s1 tok_s1  vec_s  tok_s  vec_a tok_a pred_v v r
      tl.Parallel(decode_seq,
                  decode_seq,
                  decode_seq),                # pred_s1 tok_s1 pred_s  tok_s pred_a tok_a pred_v v r
  )

  # return tl.Serial(
  #                                             #       0      1      2      3      4     5      6 7 8
  #     # Input:                                #   tok_s  tok_a tok_s1      r      v
  #     tl.Select([0, 1, 2, 0, 1, 3, 4]),       #   tok_s  tok_a tok_s1  tok_s  tok_a     r      v
  #
  #     # Encode.
  #     compress_seq,                           #   vec_s  tok_a tok_s1  tok_s  tok_a     r      v
  #     # Decode.
  #     tl.Select([0, 3, 1, 2, 3, 3]),          #   vec_s  tok_s  tok_a  tok_s1 tok_s tok_s  tok_a v r
  #     decode_seq,                             #  pred_s tok_s1  tok_a  tok_s1 tok_s tok_s  tok_a v r
  # )


@assert_shape('bld->b1d')
def PickFirst():
  """
  Select the last hidden state as the hidden representation of
  """
  def f(input):
    return jnp.expand_dims(input[:, 0, :], 1)
  return Fn('PickFirst', f)


def Squeeze(axis_to_squeeze=1):
  layer_name = f'Squeeze{axis_to_squeeze}'
  def f(x):  # pylint: disable=invalid-name
    in_rank = len(x.shape)
    if in_rank <= axis_to_squeeze:
      raise ValueError(f'Input rank ({in_rank}) must exceed the number of '
                       f'axes to keep ({axis_to_squeeze}) after squeezing.')
    return jnp.squeeze(x, axis=axis_to_squeeze)
  return Fn(layer_name, f)


def ConcatDeEntoEnDe():
  """
  Concatenate
  """
  def f(de, en):
    return jnp.hstack([en, de])
  return Fn('ConcatDeEntoEnDe', f)



def DropFirst():
  """
  Drop first in the seq.
  """
  def f(x):
    if len(x.shape) == 3:
      return x[:, 1:, :]
    elif len(x.shape) == 2:
      return x[:, 1:]
    else:
      raise ValueError("Input dimension not supported for DropFirst")
  return Fn(f'DropFirst', f)


def DropLast(mode):
  """
  Drop last in the seq.
  """
  def f(x):
    if mode == "predict":
      return x
    return x[:, :-1]
  return Fn(f'DropLast', f)


def _DecoderBlock(d_model, d_ff, n_heads,
                  dropout, dropout_shared_axes, mode, ff_activation):
  """Returns a list of layers that implements a Transformer decoder block.

  The input is an activation tensor.

  Args:
    d_model: Final dimension of tensors at most points in the model, including
        the initial embedding output.
    d_ff: Size of special dense layer in the feed-forward part of each block.
    n_heads: Number of attention heads.
    dropout: Stochastic rate (probability) for dropping an activation value
        when applying dropout within a block.
    dropout_shared_axes: Tensor axes on which to share a dropout mask.
        Sharing along batch and sequence axes (`dropout_shared_axes=(0,1)`) is
        a useful way to save memory and apply consistent masks to activation
        vectors at different sequence positions.
    mode: If `'train'`, each block will include dropout; else, it will
        pass all values through unaltered.
    ff_activation: Type of activation function at the end of each block; must
        be an activation-type subclass of `Layer`.

  Returns:
    A list of layers that maps an activation tensor to an activation tensor.
  """
  causal_attention = tl.CausalAttention(
      d_model, n_heads=n_heads, dropout=dropout, mode=mode),

  feed_forward = _FeedForwardBlock(
      d_model, d_ff, dropout, dropout_shared_axes, mode, ff_activation)

  dropout_ = tl.Dropout(
      rate=dropout, shared_axes=dropout_shared_axes, mode=mode)

  return [
      tl.Residual(
          tl.LayerNorm(),
          causal_attention,
          dropout_,
      ),
      tl.Residual(
          feed_forward
      ),
  ]


def _FeedForwardBlock(d_model, d_ff, dropout, dropout_shared_axes,
                      mode, activation):
  """Returns a list of layers implementing a feed-forward block.

  Args:
    d_model: Final dimension of tensors at most points in the model, including
        the initial embedding output.
    d_ff: Size of special dense layer in the feed-forward part of each block.
    dropout: Stochastic rate (probability) for dropping an activation value
        when applying dropout within a block.
    dropout_shared_axes: Tensor axes on which to share a dropout mask.
        Sharing along batch and sequence axes (`dropout_shared_axes=(0,1)`) is
        a useful way to save memory and apply consistent masks to activation
        vectors at different sequence positions.
    mode: If `'train'`, each block will include dropout; else, it will
        pass all values through unaltered.
    activation: Type of activation function at the end of each block; must
        be an activation-type subclass of `Layer`.

  Returns:
    A list of layers which maps vectors to vectors.
  """
  dropout_middle = tl.Dropout(
      rate=dropout, shared_axes=dropout_shared_axes, mode=mode)
  dropout_final = tl.Dropout(
      rate=dropout, shared_axes=dropout_shared_axes, mode=mode)

  return [
      tl.LayerNorm(),
      tl.Dense(d_ff),
      activation(),
      dropout_middle,
      tl.Dense(d_model),
      dropout_final,
  ]


def _EncoderBlock(d_model, d_ff, n_heads,
                  dropout, dropout_shared_axes, mode, ff_activation):
  """Returns a list of layers that implements a Transformer encoder block.

  The input to the block is a pair, (activations, mask), where the mask was
  created from the original source tokens to prevent attending to the padding
  part of the input.

  Args:
    d_model: Final dimension of tensors at most points in the model, including
        the initial embedding output.
    d_ff: Size of special dense layer in the feed-forward part of each block.
    n_heads: Number of attention heads.
    dropout: Stochastic rate (probability) for dropping an activation value
        when applying dropout within a block.
    dropout_shared_axes: Tensor axes on which to share a dropout mask.
        Sharing along batch and sequence axes (`dropout_shared_axes=(0,1)`) is
        a useful way to save memory and apply consistent masks to activation
        vectors at different sequence positions.
    mode: If `'train'`, each block will include dropout; else, it will
        pass all values through unaltered.
    ff_activation: Type of activation function at the end of each block; must
        be an activation-type subclass of `Layer`.

  Returns:
    A list of layers that maps (activations, mask) to (activations, mask).
  """
  attention = tl.Attention(
      d_model, n_heads=n_heads, dropout=dropout, mode=mode)

  feed_forward = _FeedForwardBlock(
      d_model, d_ff, dropout, dropout_shared_axes, mode, ff_activation)

  dropout_ = tl.Dropout(
      rate=dropout, shared_axes=dropout_shared_axes, mode=mode)

  return [
      tl.Residual(
          tl.LayerNorm(),
          attention,
          dropout_,
      ),
      tl.Residual(
          feed_forward
      ),
  ]
