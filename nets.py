import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import (
    Dense, 
    Conv2D,
    AveragePooling2D, 
    GlobalAveragePooling2D, 
    UpSampling2D
)


#########################
# ## SNGAN-BASED EBM ## #
#########################

# ebm architectures derived from SNGAN architectures, 
# with spectral normalization removed throughout.

# TF2 Keras reimplementation of Pytorch SN-GAN code from Mimicry Git Repo 
# https://github.com/kwotsin/mimicry
# Original Code: Copyright (c) 2020 Kwot Sin Lee under MIT License

class EBMBlock(keras.layers.Layer):
  def __init__(self, in_channels, out_channels, hidden_channels=None, downsample=False):
    super().__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels if hidden_channels is not None else out_channels
    self.learnable_sc = in_channels != out_channels or downsample
    self.downsample = downsample

    self.conv_1 = Conv2D(filters=self.hidden_channels, kernel_size=3, padding="SAME")
    self.conv_2 = Conv2D(filters=out_channels, kernel_size=3, padding="SAME")

    self.relu = keras.activations.relu

    if self.learnable_sc:
      self.sc = Conv2D(filters=out_channels, kernel_size=1, padding="VALID")
    if self.downsample:
      self.downsampling_layer = AveragePooling2D()

  def call(self, x, training=False):

    # residual layer
    h = x
    h = self.relu(h)
    h = self.conv_1(h)
    h = self.relu(h)
    h = self.conv_2(h)
    h = self.downsampling_layer(h) if self.downsample else h

    # shortcut
    y = x
    y = self.sc(y) if self.learnable_sc else y
    y = self.downsampling_layer(y) if self.downsample else y

    return h + y

class EBMBlockStem(keras.layers.Layer):
  def __init__(self, in_channels, out_channels, hidden_channels=None, downsample=False):
    super().__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels if hidden_channels is not None else out_channels
    self.learnable_sc = in_channels != out_channels or downsample
    self.downsample = downsample

    self.conv_1 = Conv2D(filters=self.hidden_channels, kernel_size=3, padding="SAME")
    self.conv_2 = Conv2D(filters=out_channels, kernel_size=3, padding="SAME")

    self.relu = keras.activations.relu

    if self.learnable_sc:
      self.sc = Conv2D(filters=out_channels, kernel_size=1, padding="VALID")
    if self.downsample:
      self.downsampling_layer = AveragePooling2D()

  def call(self, x, training=False):

    # residual layer
    h = x
    h = self.conv_1(h)
    h = self.relu(h)
    h = self.conv_2(h)
    h = self.downsampling_layer(h) if self.downsample else h

    # shortcut
    y = x
    y = self.downsampling_layer(y) if self.downsample else y
    y = self.sc(y) if self.learnable_sc else y

    return h + y

class EBMSNGAN32(keras.Model):
  def __init__(self, nch=3, ngf=256):
    super().__init__()

    self.ngf = ngf

    # Build the layers
    self.conv_1 = EBMBlockStem(nch, self.ngf, downsample=True)
    self.block2 = EBMBlock(self.ngf, self.ngf, downsample=True)
    self.block3 = EBMBlock(self.ngf, self.ngf, downsample=False)
    self.block4 = EBMBlock(self.ngf, self.ngf, downsample=False)
    self.pool_5 = GlobalAveragePooling2D()
    self.lin_5 = Dense(1, use_bias=False)

  def call(self, x, training=False):

    x = self.conv_1(x)

    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)

    x = keras.activations.relu(x)
    x = self.pool_5(x)
    x = self.lin_5(x)

    return x

class EBMSNGAN64(keras.Model):
  def __init__(self, nch=3, ngf=1024):
    super().__init__()

    self.ngf = ngf

    # Build the layers
    self.conv_1 = EBMBlockStem(nch, self.ngf >> 4, downsample=True)
    self.block2 = EBMBlock(self.ngf >> 4, self.ngf >> 3, downsample=True)
    self.block3 = EBMBlock(self.ngf >> 3, self.ngf >> 2, downsample=True)
    self.block4 = EBMBlock(self.ngf >> 2, self.ngf >> 1, downsample=True)
    self.block5 = EBMBlock(self.ngf >> 1, self.ngf, downsample=True)
    self.pool_6 = GlobalAveragePooling2D()
    self.lin_6 = Dense(1, use_bias=False)

  def call(self, x, training=False):

    x = self.conv_1(x)

    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)

    x = keras.activations.relu(x)
    x = self.pool_6(x)
    x = self.lin_6(x)

    return x

class EBMSNGAN128(keras.Model):
  def __init__(self, nch=3, ngf=1024):
    super().__init__()

    self.ngf = ngf

    # Build the layers
    self.conv_1 = EBMBlockStem(nch, self.ngf >> 4, downsample=True)
    self.block2 = EBMBlock(self.ngf >> 4, self.ngf >> 3, downsample=True)
    self.block3 = EBMBlock(self.ngf >> 3, self.ngf >> 2, downsample=True)
    self.block4 = EBMBlock(self.ngf >> 2, self.ngf >> 1, downsample=True)
    self.block5 = EBMBlock(self.ngf >> 1, self.ngf, downsample=True)
    self.block6 = EBMBlock(self.ngf, self.ngf, downsample=False)
    self.pool_7 = GlobalAveragePooling2D()
    self.lin_7 = Dense(1, use_bias=False)

  def call(self, x, training=False):

    x = self.conv_1(x)

    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)
    x = self.block6(x)

    x = keras.activations.relu(x)
    x = self.pool_7(x)
    x = self.lin_7(x)

    return x


#########################
# ## SNGAN GENERATOR ## #
#########################

# gen architectures are tf2 ports of sngan generator nets
# batch norm layers are included, but not updated during training

# TF2 Keras reimplementation of Pytorch SN-GAN code from Mimicry Git Repo 
# https://github.com/kwotsin/mimicry
# Original Code: Copyright (c) 2020 Kwot Sin Lee under MIT License

class GenBlock(keras.layers.Layer):
  def __init__(self, in_channels, out_channels, hidden_channels=None, upsample=False):
    super(GenBlock, self).__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels if hidden_channels is not None else out_channels
    self.learnable_sc = in_channels != out_channels or upsample
    self.upsample = upsample

    self.conv_1 = Conv2D(filters=self.hidden_channels, kernel_size=3, padding="SAME")
    self.conv_2 = Conv2D(filters=out_channels, kernel_size=3, padding="SAME")

    self.bn_1 = keras.layers.experimental.SyncBatchNormalization()
    self.bn_2 = keras.layers.experimental.SyncBatchNormalization()

    self.relu = keras.activations.relu

    if self.learnable_sc:
      self.sc = Conv2D(filters=out_channels, kernel_size=1, padding="VALID")
    if self.upsample:
      self.upsampling_layer = UpSampling2D(interpolation='bilinear')

  def call(self, x, training=False):

    # residual layer
    h = x
    h = self.bn_1(h, training=training)
    h = self.relu(h)
    h = self.upsampling_layer(h) if self.upsample else h
    h = self.conv_1(h)
    h = self.bn_2(h, training=training)
    h = self.relu(h)
    h = self.conv_2(h)

    # shortcut
    y = x
    y = self.upsampling_layer(y) if self.upsample else y
    y = self.sc(y) if self.learnable_sc else y

    return h + y

class GenSNGAN32(keras.Model):
  def __init__(self, nch=3, nz=128, ngf=256, bottom_width=4, out_factor=1):
    super().__init__()

    self.nz = nz
    self.ngf = ngf
    self.bottom_width = bottom_width
    self.out_factor = out_factor

    # Build the layers
    self.lin_1 = Dense((self.bottom_width**2) * self.ngf)
    self.block2 = GenBlock(self.ngf, self.ngf, upsample=True)
    self.block3 = GenBlock(self.ngf, self.ngf, upsample=True)
    self.block4 = GenBlock(self.ngf, self.ngf, upsample=True)
    self.bn_5 = keras.layers.experimental.SyncBatchNormalization()
    self.relu = keras.activations.relu
    self.conv_5 = Conv2D(filters=nch, kernel_size=3, padding="SAME")

  def generate_latent_z(self, num_ims):
    return tf.random.normal([num_ims, self.nz])  # noise sample

  def generate_images(self, num_ims):
    z = self.generate_latent_z(num_ims)
    return self.call(z)

  def call(self, x, training=False):

    h = self.lin_1(x)
    h = tf.reshape(h, (-1, self.ngf, self.bottom_width, self.bottom_width))
    h = tf.transpose(h, (0, 2, 3, 1))
    h = self.block2(h, training=training)
    h = self.block3(h, training=training)
    h = self.block4(h, training=training)
    h = self.bn_5(h, training=training)
    h = self.relu(h)
    h = self.conv_5(h)
    h = keras.activations.tanh(h)

    return self.out_factor * h

class GenSNGAN64(keras.Model):
  def __init__(self, nch=3, nz=128, ngf=1024, bottom_width=4, out_factor=1.0):
    super().__init__()

    self.nz = nz
    self.ngf = ngf
    self.bottom_width = bottom_width
    self.out_factor = out_factor

    # Build the layers
    self.lin_1 = Dense((self.bottom_width**2) * self.ngf)
    self.block2 = GenBlock(self.ngf, self.ngf >> 1, upsample=True)
    self.block3 = GenBlock(self.ngf >> 1, self.ngf >> 2, upsample=True)
    self.block4 = GenBlock(self.ngf >> 2, self.ngf >> 3, upsample=True)
    self.block5 = GenBlock(self.ngf >> 3, self.ngf >> 4, upsample=True)
    self.bn_6 = keras.layers.experimental.SyncBatchNormalization()
    self.relu = keras.activations.relu
    self.conv_6 = Conv2D(filters=nch, kernel_size=3, padding="SAME")

  def generate_latent_z(self, num_ims):
    return tf.random.normal([num_ims, self.nz])  # noise sample

  def generate_images(self, num_ims):
    z = self.generate_latent_z(num_ims)
    return self.call(z)

  def call(self, x, training=False):

    h = self.lin_1(x)
    h = tf.reshape(h, (-1, self.ngf, self.bottom_width, self.bottom_width))
    h = tf.transpose(h, (0, 2, 3, 1))
    h = self.block2(h, training=training)
    h = self.block3(h, training=training)
    h = self.block4(h, training=training)
    h = self.block5(h, training=training)
    h = self.bn_6(h, training=training)
    h = self.relu(h)
    h = self.conv_6(h)
    h = keras.activations.tanh(h)

    return self.out_factor * h

class GenSNGAN128(keras.Model):
  def __init__(self, nch=3, nz=128, ngf=1024, bottom_width=4, out_factor=1.0):
    super().__init__()

    self.nz = nz
    self.ngf = ngf
    self.bottom_width = bottom_width
    self.out_factor = out_factor

    # Build the layers
    self.lin_1 = Dense((self.bottom_width**2) * self.ngf)
    self.block2 = GenBlock(self.ngf, self.ngf, upsample=True)
    self.block3 = GenBlock(self.ngf, self.ngf >> 1, upsample=True)
    self.block4 = GenBlock(self.ngf >> 1, self.ngf >> 2, upsample=True)
    self.block5 = GenBlock(self.ngf >> 2, self.ngf >> 3, upsample=True)
    self.block6 = GenBlock(self.ngf >> 3, self.ngf >> 4, upsample=True)
    self.bn_7 = keras.layers.experimental.SyncBatchNormalization()
    self.relu = keras.activations.relu
    self.conv_7 = Conv2D(filters=nch, kernel_size=3, padding="SAME")

  def generate_latent_z(self, num_ims):
    return tf.random.normal([num_ims, self.nz])  # noise sample

  def generate_images(self, num_ims):
    z = self.generate_latent_z(num_ims)
    return self.call(z)

  def call(self, x, training=False):

    h = self.lin_1(x)
    h = tf.reshape(h, (-1, self.ngf, self.bottom_width, self.bottom_width))
    h = tf.transpose(h, (0, 2, 3, 1))
    h = self.block2(h, training=training)
    h = self.block3(h, training=training)
    h = self.block4(h, training=training)
    h = self.block5(h, training=training)
    h = self.block6(h, training=training)
    h = self.bn_7(h, training=training)
    h = self.relu(h)
    h = self.conv_7(h)
    h = keras.activations.tanh(h)

    return self.out_factor * h


########################################
# ## INF AND GEN SNGAN FOR RETROFIT ## #
########################################

class GenBlockStem(keras.layers.Layer):
  def __init__(self, in_channels, out_channels, hidden_channels=None, upsample=False):
    super().__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels if hidden_channels is not None else out_channels
    self.learnable_sc = in_channels != out_channels or upsample
    self.upsample = upsample

    self.conv_1 = Conv2D(filters=self.hidden_channels, kernel_size=3, padding="SAME")
    self.conv_2 = Conv2D(filters=out_channels, kernel_size=3, padding="SAME")

    self.bn_2 = keras.layers.experimental.SyncBatchNormalization()
    self.relu = keras.activations.relu

    if self.learnable_sc:
      self.sc = Conv2D(filters=out_channels, kernel_size=1, padding="VALID")
    if self.upsample:
      self.upsampling_layer = UpSampling2D(interpolation='bilinear')

  def call(self, x, training=False):

    # residual layer
    h = x
    h = self.upsampling_layer(h) if self.upsample else h
    h = self.conv_1(h)
    h = self.bn_2(h, training=training)
    h = self.relu(h)
    h = self.conv_2(h)

    # shortcut
    y = x
    y = self.upsampling_layer(y) if self.upsample else y
    y = self.sc(y) if self.learnable_sc else y

    return h + y

# similar to GenSNGAN32, except latent space has dimension [16, 16, 1]
class GenDecoder32(keras.Model):
  def __init__(self, nch=3, nz=[16, 16, 1], ngf=256, out_factor=1):
    super().__init__()

    self.nz = nz
    self.ngf = ngf
    self.nch = nch
    self.out_factor = out_factor

    # Build the layers
    self.block1 = GenBlockStem(self.nch, self.ngf, upsample=False)
    self.block2 = GenBlock(self.ngf, self.ngf, upsample=True)
    self.block3 = GenBlock(self.ngf, self.ngf, upsample=False)
    self.block4 = GenBlock(self.ngf, self.ngf, upsample=False)
    self.bn_5 = keras.layers.experimental.SyncBatchNormalization()
    self.relu = keras.activations.relu
    self.conv_5 = Conv2D(filters=nch, kernel_size=3, padding="SAME")

  def generate_latent_z(self, num_ims):
    return tf.random.normal([num_ims] + self.nz)  # noise sample

  def generate_images(self, num_ims):
    z = self.generate_latent_z(num_ims)
    return self.call(z)

  def call(self, x, training=False):

    h = self.block1(x, training=training)
    h = self.block2(h, training=training)
    h = self.block3(h, training=training)
    h = self.block4(h, training=training)
    h = self.bn_5(h, training=training)
    h = self.relu(h)
    h = self.conv_5(h)
    h = keras.activations.tanh(h)

    return self.out_factor * h

# inference net where latent space has dimension [16, 16, 1]
class InfEncoder32(keras.Model):
  def __init__(self, nch=3, ngf=256, z_sz=[16, 16, 1]):
    super().__init__()

    self.ngf = ngf
    self.nch = nch
    self.z_sz = z_sz

    # Build the layers
    self.block1 = EBMBlockStem(self.nch, self.ngf, downsample=False)
    self.block2 = EBMBlock(self.ngf, self.ngf, downsample=True)
    self.block3 = EBMBlock(self.ngf, self.ngf, downsample=False)
    self.block4 = EBMBlock(self.ngf, self.ngf, downsample=False)
    self.conv2latent = Conv2D(1, 3, padding="SAME")

  def call(self, x, training=False, normalize=True, eps=1e-6):

    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)

    x = keras.activations.relu(x)
    x = self.conv2latent(x)

    if normalize:
      # project inferred states to sphere with radius sqrt(latent_dim)
      x_flat = tf.reshape(x, (tf.shape(x)[0], -1))
      x_flat_norm = tf.sqrt(tf.math.reduce_sum(tf.math.pow(x_flat, 2.0), axis=1))
      x_flat_norm = tf.reshape(x_flat_norm + eps, [-1, 1, 1, 1])
      proj_radius = tf.math.sqrt(tf.cast(tf.shape(x_flat)[1], tf.float32))
      x = proj_radius * x / x_flat_norm

    return x


##################################
# ## FUNCTION TO GET NETWORKS ## #
##################################

def create_net(config, net_type):

  # ebm networks
  if net_type == 'ebm_sngan':
    if config['image_dims'][0] == 32:
      return EBMSNGAN32(nch=config['image_dims'][2])
    elif config['image_dims'][0] == 64:
      return EBMSNGAN64(nch=config['image_dims'][2])
    elif config['image_dims'][0] == 128:
      return EBMSNGAN128(nch=config['image_dims'][2], ngf=config['ebm_ngf'])
    else:
      raise ValueError('Invalid image_dims for ebm_sngan')

  # generator networks
  if net_type == 'gen_sngan':
    if config['image_dims'][0] == 32:
      return GenSNGAN32(nch=config['image_dims'][2], out_factor=config['gen_factor'])
    elif config['image_dims'][0] == 64:
      return GenSNGAN64(nch=config['image_dims'][2], out_factor=config['gen_factor'])
    elif config['image_dims'][0] == 128:
      return GenSNGAN128(nch=config['image_dims'][2], 
                         ngf=config['gen_ngf'],
                         out_factor=config['gen_factor'])
    else:
      raise ValueError('Invalid image_dims for sngan gen net')
  elif net_type == 'gen_decoder':
    return GenDecoder32(nch=config['image_dims'][2], out_factor=config['gen_factor'])

  # inference networks
  if net_type == 'inf_encoder':
    return InfEncoder32(nch=config['image_dims'][2])

  # raise error if no net is returned by this point
  raise ValueError('Invalid "net_type".')
