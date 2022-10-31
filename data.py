import tensorflow as tf
import tensorflow_datasets as tfds


# function to set up dataset
def get_dataset(data_type, 
                batch_size,
                image_dims,
                data_dir,
                shuffle=True, 
                repeat=True, 
                split='train',
                get_label=False,
                random_crop=False
              ):

  assert image_dims[0] == image_dims[1], 'Only square image sizes are supported.'

  # load tfrecords
  dataset = tfds.load(
                name=data_type, 
                split=split, 
                data_dir=data_dir, 
                download=False, 
                try_gcs=False, 
                with_info=False, 
                as_supervised=False, 
                shuffle_files=shuffle
            )

  def transform(features, scale_range=[1., 1.], aspect_range=[1., 1.], data_flip=False):
    # get image, cast to float, scale to [-1, 1] pixel range
    image = features['image']
    image = tf.cast(image, tf.float32)

    # scale from [0, 255] range to [-1, 1] range
    image = 2 * (image / 255.0) - 1

    if random_crop:
      # change aspect and select random size and location crop
      image = resize_random_crop(image, image_dims, scale_range, aspect_range)
    elif image.shape[0:2] != image_dims[0:2]:
      # center crop and resize
      image = center_crop(image, image_dims[0])

    # left-right random flip
    if data_flip:
      image = tf.image.random_flip_left_right(image)

    # return image with or without label
    if not get_label:
      return image
    else:
      label = features['label']
      return image, label

  dataset = dataset.map(transform)

  if shuffle:
    # shuffle with buffer size
    dataset = dataset.shuffle(10000)
  if repeat:
    # infinite data loop
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
  else:
    dataset = dataset.batch(batch_size, drop_remainder=True)

  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

  return dataset

def resize_random_crop(image, target_dims, scale_range=[1., 1.], aspect_range=[1., 1.]):
  # function to replicate torchvision.transforms.RandomResizedCrop

  # get rescaled dimensions
  width, height = tf.shape(image)[0], tf.shape(image)[1]

  # get aspect of warped image (no warping if aspect range is [1, 1])
  aspect_log = tf.math.log(tf.constant(aspect_range))
  target_aspect = tf.math.exp(tf.random.uniform(shape=[1], minval=aspect_log[0], maxval=aspect_log[1]))
  # get dimensions of warped orig image and warped image at target scale
  width_new = tf.cast(width, tf.float32) * tf.sqrt(target_aspect)
  height_new = tf.cast(height, tf.float32) / tf.sqrt(target_aspect)
  width_resize = (target_dims[0] / tf.math.minimum(width_new, height_new)) * width_new
  height_resize = (target_dims[1] / tf.math.minimum(width_new, height_new)) * height_new
  width_resize = tf.math.maximum(tf.cast(width_resize, dtype=tf.int32), tf.constant(target_dims[0]))
  height_resize = tf.math.maximum(tf.cast(height_resize, dtype=tf.int32), tf.constant(target_dims[1]))

  # rescale image according to target aspect to be a valid size for cropping
  image = tf.image.resize(image, tf.concat([width_resize, height_resize], 0), antialias=True)
  # select the scale of the patch to get from resized image
  scale = tf.math.sqrt(tf.random.uniform(shape=[1], minval=scale_range[0], maxval=scale_range[1]))
  scaled_dims = scale * tf.constant([target_dims[0], target_dims[1]], dtype=tf.float32)
  # get the patch according to the scaled dimensions
  crop_dims = tf.concat((tf.cast(scaled_dims, tf.int32), tf.constant([target_dims[2]])), 0)
  image = tf.image.random_crop(image, crop_dims)

  # resize patch to the desired size
  # if scale_range is [1, 1], this will return the patch with no size change
  image = tf.image.resize(image, (target_dims[0], target_dims[1]), antialias=True)

  return image

def center_crop(image, target_dim):
  # get rescaled dimensions
  height = tf.shape(image)[0]
  width = tf.shape(image)[1]
  # center crop
  if height > width:
    image = tf.image.crop_to_bounding_box(image, (height - width) // 2, 0, width, width)
  elif width > height:
    image = tf.image.crop_to_bounding_box(image, 0, (width - height) // 2, height, height)
  # resize to the desired size
  image = tf.image.resize(image, (target_dim, target_dim), antialias=True)

  return image
