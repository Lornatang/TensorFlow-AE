# Copyright 2019 ChangyuLiu Authors. All Rights Reserved.
#
#           Licensed under the MIT License.
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at 
#         https://opensource.org/licenses/MIT
# ==============================================================================

"""Generate optim loss and Discriminate optim loss"""

import tensorflow as tf


def ae_loss(noise, fake_output):
  """  Automatic coding loss function

  Args:
    noise: .
    fake_output: generate pic for use encoder model

  Returns:
    tf.reduce_mean.

  """
  return tf.reduce_mean(tf.square(noise - fake_output))


def decoder_optimizer():
  """ The training generator optimizes the network.

  Returns:
    tf.optimizers.Adam.

  """
  return tf.optimizers.Adam()


def encoder_optimizer():
  """ The training discriminator optimizes the network.

  Returns:
    tf.optimizers.Adam.

  """
  return tf.optimizers.Adam()
