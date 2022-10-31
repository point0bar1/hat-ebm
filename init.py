import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

from nets import create_net
from data import get_dataset
from utils import download_blob

import os
import pickle


def initialize_strategy(device_type):
  if device_type == 'tpu':
    # Set up TPU Distribution (set to run from Cloud TPU VM)
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
    #tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))
    strategy = tf.distribute.TPUStrategy(resolver)
  elif device_type == 'gpu':
    # set up gpus for rtx compatibility
    def init_tf2(tf_eager=False, memory_growth=True, disable_meta_optimizer=True):
      tf.config.set_soft_device_placement(True)
      gpus = tf.config.experimental.list_physical_devices('GPU')
      if memory_growth and gpus:
        for gpu in gpus:
          # rtx needs memory growth for multi-gpu
          # https://github.com/tensorflow/tensorflow/issues/29632
          tf.config.experimental.set_memory_growth(gpu, True)
      tf.config.run_functions_eagerly(tf_eager)
      tf.config.optimizer.set_experimental_options({'disable_meta_optimizer': disable_meta_optimizer})
    init_tf2()
    # set up GPU Distribution (also works for single GPU)
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    # suppress retracing warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  elif device_type == 'cpu':
    strategy = tf.distribute.OneDeviceStrategy('CPU:0')
  else:
    raise ValueError('Invalid device_type.')
  return strategy


def initialize_net_and_optim(
  config, 
  strategy, 
  net_type, 
  weight_path=None, 
  optim_type=None, 
  optim_lr_info=None, 
  optim_decay=0.0, 
  optim_weights=None,
  optim_state_dims=None
):

  # Create the model, optimizer and metrics inside strategy scope, so that the
  # variables can be mirrored on each device.
  with strategy.scope():
    # set up ebm and optinally load weights
    net = create_net(config, net_type)
    if weight_path is not None:
      net.load_weights(weight_path)

    # set up ebm optimizer
    if optim_type is not None:
      # set up ebm optimizer schedule
      lr_schedule = StepScheduleLR(optim_lr_info)

      # set up ebm optim
      if optim_type == 'adam':
        optim = Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999)
      elif optim_type == 'sgd':
        optim = SGD(learning_rate=lr_schedule)
      elif optim_type == 'rms':
        optim = RMSprop(learning_rate=lr_schedule, momentum=0.1)
      else:
        raise RuntimeError('Invalid ebm_optim_type')

      # moving average for weights
      optim = tfa.optimizers.MovingAverage(optim, average_decay=optim_decay)

    else:
      optim = None

  # function to download optim weights saved on cloud
  def download_optim(file_name):
    exp_folder = os.path.join(config['exp_dir'], config['exp_name'])

    # download persistent ims from cloud
    temp_optim_path = os.path.join(exp_folder, 'checkpoints/optim_download.ckpt')
    download_blob(config['gs_path'], file_name, temp_optim_path)
    optim_weights = pickle.load(open(temp_optim_path, 'rb'))
    # remove to save space
    os.remove(temp_optim_path)

    return optim_weights

  # optim weight initialization (optional)
  if optim_weights is not None: 
    # dummy function to initialize optim
    @tf.function
    def initialize_optim():
      _ = net(tf.random.normal(shape=[3]+optim_state_dims))
      null_grads = [tf.zeros_like(var) for var in net.trainable_variables]
      optim.apply_gradients(list(zip(null_grads, net.trainable_variables)))
    strategy.run(initialize_optim)

    # load optim weights
    if config['gs_path'] is not None:
      # load from cloud storage
      optim_weights = download_optim(optim_weights)
    else:
      # load from local storage
      optim_weights = pickle.load(open(optim_weights, 'rb'))
    optim.set_weights(optim_weights)

  return net, optim


def initialize_persistent(
  config, 
  strategy,
  init_type,  # 'data', or 'noise'
  persistent_size,
  state_dims=None,
  persistent_init_scale=1.0,
  persistent_path=None,
  z_bank=None,
  gen=None
):

  print('Initializing persistent_states...')

  # same number of states on each device
  per_replica_persistent_size = persistent_size // strategy.num_replicas_in_sync
  # ensure batch size divides number of states per device
  per_replica_batch_size = config['batch_size'] // strategy.num_replicas_in_sync
  per_replica_batches = per_replica_persistent_size // per_replica_batch_size
  per_replica_persistent_size = per_replica_batch_size * per_replica_batches
  # final size of tensor bank
  persistent_size = strategy.num_replicas_in_sync * per_replica_persistent_size
  persistent_tensor_size = [persistent_size] + state_dims

  # function to download existing persistent states (optional)
  def download_persistent():
    exp_folder = os.path.join(config['exp_dir'], config['exp_name'])
    # download persistent ims from cloud
    temp_persistent_path = os.path.join(exp_folder, 'checkpoints/persistent.ckpt')
    download_blob(config['gs_path'], persistent_path, temp_persistent_path)
    persistent_tensor = pickle.load(open(temp_persistent_path, 'rb'))
    # remove to save space
    os.remove(temp_persistent_path)

    return persistent_tensor

  if persistent_path is not None:
    # load saved persistent states (optional, for restarting exp)
    if config['gs_path'] is not None:
      # load from cloud
      persistent_tensor_init = download_persistent()
    else:
      # load from local file
      persistent_tensor_init = pickle.load(open(persistent_path, 'rb'))
  elif z_bank is not None:
    # gather latent states to initialize images from generator output
    persistent_tensor_init = strategy.gather(z_bank, 0)
  elif init_type == 'data':
    # iterator to initialize states from data
    dataset_mcmc_init = strategy.distribute_datasets_from_function(
      lambda _: get_dataset(
        config['data_type'],
        per_replica_persistent_size,
        config['image_dims'],
        'gs://' + config['gs_path'],
        random_crop=config['random_crop']
      )
    )
    persistent_tensor_init = next(iter(dataset_mcmc_init))
    persistent_tensor_init = strategy.gather(persistent_tensor_init, 0)
  elif init_type == 'noise':
    # initialize from gaussian noise distribution
    persistent_tensor_init = persistent_init_scale * tf.random.normal(persistent_tensor_size)
  else:
    raise ValueError('Invalid "init_type".')

  def initialize_persistent_states(ctx):
    # select cpu states according to device ID
    persistent_tensor_device = \
        persistent_tensor_init[(per_replica_persistent_size * ctx.replica_id_in_sync_group):
                               (per_replica_persistent_size * (ctx.replica_id_in_sync_group + 1))]
    if persistent_path is None and z_bank is not None:
      # pass through generator network to get initial paired images
      out_tensor = tf.zeros(shape=[per_replica_persistent_size] + config['image_dims'])
      for batch_ind in tf.range(per_replica_persistent_size // per_replica_batch_size):
        batch_range = tf.range(per_replica_batch_size*batch_ind, per_replica_batch_size*(batch_ind+1))
        gen_batch = gen(tf.gather(persistent_tensor_device, batch_range))
        batch_range = tf.reshape(batch_range, shape=[-1, 1])
        out_tensor = tf.tensor_scatter_nd_update(out_tensor, batch_range, gen_batch)
      return out_tensor
    else:
      # initialize from tensor state on cpu
      return persistent_tensor_device
  persistent_states = strategy.experimental_distribute_values_from_function(initialize_persistent_states)

  return persistent_states


def determinism_test(strategy, net, state_dims):
  # test deterministic output of network, up to variation from cuDNN randomization
  with strategy.scope():
    z_test = tf.random.normal(shape=[3]+state_dims)
    z_out_1 = net(z_test, training=False)
    z_out_2 = net(z_test[0:2], training=False)
  z_out_1 = strategy.gather(z_out_1, axis=0)
  z_out_2 = strategy.gather(z_out_2, axis=0)

  max_abs_diff = tf.math.reduce_max(tf.math.abs(z_out_1[0] - z_out_2[0]))
  max_abs_scale = tf.math.reduce_max(tf.math.abs(z_out_1[0]))

  print('Determinism Test (if deterministic, should be very close to 0): ', 
        max_abs_diff / max_abs_scale)

# schedule for optimizer learning rate
# expects list in form [[lr_1, step_1], [lr_2, step_2], ... , [lr_n, step_n]]
# where lr_j is the learning rate, step_j is the train step where lr_j is first used
class StepScheduleLR(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, lr_info):
    super().__init__()

    self.lr_steps = tf.constant([lr_pair[0] for lr_pair in lr_info])
    self.lr_thresholds = tf.constant([lr_pair[1] for lr_pair in lr_info])
    self.lr_thresholds = tf.cast(self.lr_thresholds, dtype=tf.float32)

  def __call__(self, step):
    lr_ind = tf.math.reduce_max(tf.where(tf.cast(step, tf.float32) >= self.lr_thresholds))
    return self.lr_steps[lr_ind]
