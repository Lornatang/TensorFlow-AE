# Copyright 2019 ChangyuLiu Authors. All Rights Reserved.
#
#           Licensed under the MIT License.
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#         https://opensource.org/licenses/MIT
# ==============================================================================

"""The training loop begins with generator receiving a random seed as input.
   That seed is used to produce an image.
   The discriminator is then used to classify real images (drawn from the training set)
   and fakes images (produced by the generator).
   The loss is calculated for each of these models,
   and the gradients are used to update the generator and discriminator.
"""

from data.datasets import load_data
from model.decoder import make_decoder_model
from model.encoder import make_encoder_model
from util.loss_and_optim import ae_loss, encoder_optimizer, decoder_optimizer
from util.save_checkpoints import save_checkpoints
from util.generate_and_save_images import generate_and_save_images

import tensorflow as tf
import time
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=50, type=int,
                    help='Epochs for training.')
args = parser.parse_args()
print(args)

# define model save path
save_path = './training_checkpoints'

BUFFER_SIZE = 60000
BATCH_SIZE = 128

num_examples_to_generate = 16
noise_dim = 64

# create dir
if not os.path.exists(save_path):
  os.makedirs(save_path)

# define random noise
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# load dataset
train_dataset = load_data(BUFFER_SIZE, BATCH_SIZE)

# load network and optim paras
decoder = make_decoder_model()
decoder_optimizer = decoder_optimizer()

encoder = make_encoder_model()
encoder_optimizer = encoder_optimizer()

checkpoint_dir, checkpoint, checkpoint_prefix = save_checkpoints(decoder,
                                                                 encoder,
                                                                 decoder_optimizer,
                                                                 encoder_optimizer,
                                                                 save_path)


# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
  """ break it down into training steps.

  Args:
    images: input images.

  """
  with tf.GradientTape() as de_tape, tf.GradientTape() as en_tape:
    z = encoder(images)
    _x = decoder(z)

    ae_of_loss = ae_loss(images, _x)

  gradients_of_decoder = de_tape.gradient(ae_of_loss,
                                          decoder.trainable_variables)
  gradients_of_encoder = en_tape.gradient(ae_of_loss,
                                          encoder.trainable_variables)

  decoder_optimizer.apply_gradients(
    zip(gradients_of_decoder, decoder.trainable_variables))
  encoder_optimizer.apply_gradients(
    zip(gradients_of_encoder, encoder.trainable_variables))


def train(dataset, epochs):
  """ train op

  Args:
    dataset: mnist dataset or cifar10 dataset.
    epochs: number of iterative training.

  """
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as we go
    generate_and_save_images(decoder,
                             epoch + 1,
                             seed,
                             save_path)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

    print(f'Time for epoch {epoch + 1} is {time.time() - start:.3f} sec.')

  # Generate after the final epoch
  generate_and_save_images(decoder,
                           epochs,
                           seed,
                           save_path)


if __name__ == '__main__':
  train(train_dataset, args.epochs)
