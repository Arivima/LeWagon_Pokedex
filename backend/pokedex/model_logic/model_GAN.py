import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

import tensorflow as tf
from tensorflow.keras import preprocessing
from keras.models import Sequential
from keras.layers import Conv2D,Dropout,Dense,Flatten,Conv2DTranspose,BatchNormalization,LeakyReLU,Reshape
import numpy as np
import matplotlib.pyplot as plt
import os
# from zipfile import ZipFile
# import imageio
from tqdm import tqdm
# import time




def DiffAugment(x, policy='', channels_first=False):
    if policy:
        if channels_first:
            x = tf.transpose(x, [0, 2, 3, 1])
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if channels_first:
            x = tf.transpose(x, [0, 3, 1, 2])
    return x


def rand_brightness(x):
    magnitude = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) - 0.5
    x = x + magnitude
    return x


def rand_saturation(x):
    magnitude = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) * 2
    x_mean = tf.reduce_mean(x, axis=3, keepdims=True)
    x = (x - x_mean) * magnitude + x_mean
    return x


def rand_contrast(x):
    magnitude = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) + 0.5
    x_mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
    x = (x - x_mean) * magnitude + x_mean
    return x


def rand_translation(x, ratio=0.125):
    batch_size = tf.shape(x)[0]
    image_size = tf.shape(x)[1:3]
    shift = tf.cast(tf.cast(image_size, tf.float32) * ratio + 0.5, tf.int32)
    translation_x = tf.random.uniform([batch_size, 1], -shift[0], shift[0] + 1, dtype=tf.int32)
    translation_y = tf.random.uniform([batch_size, 1], -shift[1], shift[1] + 1, dtype=tf.int32)
    grid_x = tf.clip_by_value(tf.expand_dims(tf.range(image_size[0], dtype=tf.int32), 0) + translation_x + 1, 0,
                              image_size[0] + 1)
    grid_y = tf.clip_by_value(tf.expand_dims(tf.range(image_size[1], dtype=tf.int32), 0) + translation_y + 1, 0,
                              image_size[1] + 1)
    x = tf.gather_nd(tf.pad(x, [[0, 0], [1, 1], [0, 0], [0, 0]]), tf.expand_dims(grid_x, -1), batch_dims=1)
    x = tf.transpose(tf.gather_nd(tf.pad(tf.transpose(x, [0, 2, 1, 3]), [[0, 0], [1, 1], [0, 0], [0, 0]]),
                                  tf.expand_dims(grid_y, -1), batch_dims=1), [0, 2, 1, 3])
    return x


def rand_cutout(x, ratio=0.5):
    batch_size = tf.shape(x)[0]
    image_size = tf.shape(x)[1:3]
    cutout_size = tf.cast(tf.cast(image_size, tf.float32) * ratio + 0.5, tf.int32)
    offset_x = tf.random.uniform([tf.shape(x)[0], 1, 1], maxval=image_size[0] + (1 - cutout_size[0] % 2),
                                 dtype=tf.int32)
    offset_y = tf.random.uniform([tf.shape(x)[0], 1, 1], maxval=image_size[1] + (1 - cutout_size[1] % 2),
                                 dtype=tf.int32)
    grid_batch, grid_x, grid_y = tf.meshgrid(tf.range(batch_size, dtype=tf.int32),
                                             tf.range(cutout_size[0], dtype=tf.int32),
                                             tf.range(cutout_size[1], dtype=tf.int32), indexing='ij')
    cutout_grid = tf.stack(
        [grid_batch, grid_x + offset_x - cutout_size[0] // 2, grid_y + offset_y - cutout_size[1] // 2], axis=-1)
    mask_shape = tf.stack([batch_size, image_size[0], image_size[1]])
    cutout_grid = tf.maximum(cutout_grid, 0)
    cutout_grid = tf.minimum(cutout_grid, tf.reshape(mask_shape - 1, [1, 1, 1, 3]))
    mask = tf.maximum(
        1 - tf.scatter_nd(cutout_grid, tf.ones([batch_size, cutout_size[0], cutout_size[1]], dtype=tf.float32),
                          mask_shape), 0)
    x = x * tf.expand_dims(mask, axis=3)
    return x

##########################################################
#VARIABLES :

AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}


# tf.keras.utils.set_random_seed(7)
# batch_size = 32
# path = "/kaggle/input/all-data-pokemon/all_data_name"
# trained_models_folder = "/kaggle/working/models"
# generated_images_folder = "/kaggle/working/images"


##########################################################
def gan_process(path,batch_size):
    dataset = preprocessing.image_dataset_from_directory(
        path, label_mode=None, image_size=(128, 128), batch_size=batch_size
    )
    dataset = dataset.map(lambda x: (x - 127.5) / 127.5)
    return dataset

# dataset = gan_process(path)

def initialize_discriminator():
    discriminator = Sequential(
    [
        tf.keras.Input(shape=(128, 128, 3)),
        tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding="same",
                      kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False),
        tf.keras.layers.LeakyReLU(0.2),

        tf.keras.layers.Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding="same",
                      kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02),
        tf.keras.layers.LeakyReLU(0.2),

        tf.keras.layers.Conv2D(256, kernel_size=(5, 5), strides=(2, 2), padding="same",
                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                            use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02),
        tf.keras.layers.LeakyReLU(0.2),

        tf.keras.layers.Conv2D(512, kernel_size=(5, 5), strides=(2, 2), padding="same",
                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                            use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02),
        tf.keras.layers.LeakyReLU(0.2),

        tf.keras.layers.Conv2D(1024, kernel_size=(5, 5), strides=(2, 2), padding="same",
                               kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                               use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02),
        tf.keras.layers.LeakyReLU(0.2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ],
    name="discriminator",
)
    return discriminator


def initialize_generator(latent_dim = 100):
    generator = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(latent_dim,)),
        tf.keras.layers.Dense(8 * 8 * 1024),
        tf.keras.layers.Reshape((8, 8, 1024)),
        tf.keras.layers.Conv2DTranspose(512, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                     kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                                     use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2DTranspose(256, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                     kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                                     use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2DTranspose(128, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                     kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                                     use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2DTranspose(64, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                                        use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2D(3,  kernel_size=(5, 5), strides=(1, 1), padding='same',
                                     kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                                     use_bias=False, activation='tanh')
    ],
    name="generator",
)
    return generator


# random_noise = tf.random.normal([1, latent_dim])
# generated_image = generator(random_noise, training=False)
# plt.imshow(generated_image[0])
# plt.show()


# decision = discriminator(generated_image)
# print('decision', decision)
# binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()

def generator_loss(label, fake_output):
    binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()
    gen_loss = binary_cross_entropy(label, fake_output)
    return gen_loss

def discriminator_loss(label, output):
    binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()
    disc_loss = binary_cross_entropy(label, output)
    return disc_loss

def initialize_gen_optimizer():
    generator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
    return generator_optimizer
def initialize_disc_optimizer():
    discriminator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
    return discriminator_optimizer

@tf.function
def train_step(images,generator,discriminator,latent_dim,batch_size, discriminator_optimizer, generator_optimizer):
    noise = tf.random.normal([batch_size, latent_dim])
    images = DiffAugment(images, policy='color,translation,cutout')

    with tf.GradientTape() as disc_tape1:
        generated_images = generator(noise, training=True)
        generated_images = DiffAugment(generated_images,policy='color,translation,cutout')

        real_output = discriminator(images, training=True)
        real_targets = tf.ones_like(real_output)
        disc_loss1 = discriminator_loss(real_targets, real_output)

    gradients_disc1 = disc_tape1.gradient(disc_loss1, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_disc1, discriminator.trainable_variables))

    with tf.GradientTape() as disc_tape2:
        fake_output = discriminator(generated_images, training=True)
        fake_targets = tf.zeros_like(fake_output)
        disc_loss2 = discriminator_loss(fake_targets, fake_output)

    gradients_disc2 = disc_tape2.gradient(disc_loss2, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_disc2, discriminator.trainable_variables))

    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        generated_images = DiffAugment(generated_images, policy='color,translation,cutout')
        fake_output = discriminator(generated_images, training=True)
        real_targets = tf.ones_like(fake_output)
        gen_loss = generator_loss(real_targets, fake_output)

    gradients_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))

    return disc_loss1 + disc_loss2, gen_loss

# latent_dim =100
# seed = tf.random.normal([25, latent_dim])
# disc_losses = []
# gen_losses = []
def train_gan(dataset, epochs,trained_models_folder, generated_images_folder,seed,batch_size,latent_dim,AUGMENT_FNS):
    generator = initialize_generator()
    discriminator = initialize_discriminator()
    generate_and_save_images(generator, 0, seed, generated_images_folder)
    discriminator.save(os.path.join(trained_models_folder,"Discriminator_epoch_0.h5"))
    generator.save(os.path.join(trained_models_folder,"Generator_epoch_0.h5"))
    disc_losses = []
    gen_losses = []
    discriminator_optimizer = initialize_disc_optimizer()
    generator_optimizer = initialize_gen_optimizer()
    for epoch in range(epochs):
        disc_loss = gen_loss = 0
        print('Currently training on epoch {} (out of {}).'.format(epoch+1, epochs))
        for image_batch in tqdm(dataset):
            losses = train_step(image_batch,generator,discriminator,latent_dim,batch_size, discriminator_optimizer, generator_optimizer)
            disc_loss += losses[0]
            gen_loss += losses[1]

        generate_and_save_images(generator, epoch+1, seed, generated_images_folder)
        gen_losses.append(gen_loss.numpy())
        disc_losses.append(disc_loss.numpy())

        if epoch % 100 == 0:
            discriminator.save(os.path.join(trained_models_folder,f"Discriminator_epoch_{epoch}.h5"))
            generator.save(os.path.join(trained_models_folder,f"Generator_epoch_{epoch}.h5"))


    generate_and_save_images(generator, epochs, seed,generated_images_folder)
    discriminator.save(os.path.join(trained_models_folder,f"Discriminator_epoch_{epochs}.h5"))
    generator.save(os.path.join(trained_models_folder,f"Generator_epoch_{epochs}.h5"))


def generate_and_save_images(model, epoch, seed, generated_images_folder, dim =(5, 5), figsize=(5, 5)):
    generated_images = model(seed)
    generated_images *= 255
    generated_images.numpy()
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        img = tf.keras.preprocessing.image.array_to_img(generated_images[i])
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(generated_images_folder,f'generated_image_epoch_{epoch}.png'))
    plt.close()


# train_gan(dataset,1000)

# plt.figure()
# plt.plot(disc_losses, label='Discriminator Loss')
# plt.plot(gen_losses, label='Generator Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig(generated_images_folder + 'losses.png')
# plt.close()
