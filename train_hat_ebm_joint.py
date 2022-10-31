# train Hat EBM jointly in latent and residual image space

import os
from time import time
from datetime import datetime
import importlib

import tensorflow as tf

from utils import setup_exp, save_model, plot_ims, plot_diagnostics
from init import initialize_strategy, initialize_net_and_optim, determinism_test
from data import get_dataset
from ebm_utils import make_langevin_update, update_ebm

import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    'config_name', 
    default='configs_joint/cifar10_retrofit.py', 
    help='Name of config file.'
)
args = parser.parse_args()


###############
# ## SETUP ## #
###############

# get experiment config
config_module = importlib.import_module(args.config_name.replace('/', '.')[:-3])
config = config_module.config

# give exp_name unique timestamp identifier
time_str = datetime.now().strftime('%y-%m-%d-%H-%M-%S')
config['exp_name'] = config['exp_name'] + '_' + time_str

# setup folders, save code
setup_exp(os.path.join(config['exp_dir'], config['exp_name']), 
          ['checkpoints', 'shortrun', 'longrun', 'fid_images', 'numpy', 'plots'], 
          [code_file for code_file in
            [
              os.path.basename(__file__), 
              'nets.py', 
              'utils.py', 
              'data.py', 
              'init.py', 
              'ebm_utils.py',
              args.config_name
            ]
          ],
          config['gs_path'], config['save_to_cloud'])

# initialize distribution strategy
strategy = initialize_strategy(config['device_type'])


##################################################
# ## INITIALIZE NETS, DATA, PERSISTENT STATES ## #
##################################################

# load ebm and optim
ebm, ebm_optim = \
  initialize_net_and_optim(
    config, 
    strategy, 
    config['ebm_type'],
    optim_type=config['ebm_optim_type'],
    optim_lr_info=config['ebm_lr_info'],
    optim_decay=config['ebm_opt_decay']
  )
# test deterministic ouput
determinism_test(strategy, ebm, config['image_dims'])

# load pretrained generator
gen, _ = \
  initialize_net_and_optim(
    config, 
    strategy, 
    config['gen_type'],
    weight_path=config['gen_weights']
  )
# test deterministic ouput
determinism_test(strategy, gen, config['state_dims'])


# Calculate per replica batch size, and distribute the datasets
print('Importing data...')
num_reps = strategy.num_replicas_in_sync
per_replica_batch_size = config['batch_size'] // num_reps
tpu_latent_size = [per_replica_batch_size] + config['state_dims']
tpu_image_size = [per_replica_batch_size] + config['image_dims']

# initialize data
train_dataset = strategy.distribute_datasets_from_function(
  lambda _: get_dataset(
    config['data_type'],
    per_replica_batch_size,
    config['image_dims'],
    'gs://' + config['gs_path'] if config['gs_path'] is not None else config['data_path'],
    random_crop=config['random_crop']
  )
)
train_iterator = iter(train_dataset)

# plot example of data images
data_samples = strategy.gather(next(train_iterator), axis=0)
plot_ims(
  os.path.join(config['exp_dir'], config['exp_name'], 'shortrun/data.pdf'), 
  data_samples[0:per_replica_batch_size]
)


with strategy.scope():
  # metrics
  ebm_loss_metric = tf.keras.metrics.Mean('ebm_loss', dtype=tf.float32)
  latent_grad_norm_metric = tf.keras.metrics.Mean('latent_grad_norm', dtype=tf.float32)
  res_grad_norm_metric = tf.keras.metrics.Mean('res_grad_norm', dtype=tf.float32)
  z_norm_metric = tf.keras.metrics.Mean('z_norm', dtype=tf.float32)


###########################
# ## TF GRAPH BUILDERS ## #
###########################

# graph for langevin sampling
langevin_update = make_langevin_update(config, ebm, gen)

@tf.function
def step_fn(images_data_in):

  # initial samples for mcmc
  lats_in = config['init_scale_latent'] * tf.random.normal(shape=tpu_latent_size)
  res_in = config['init_scale_res'] * tf.random.normal(shape=tpu_image_size)

  # langevin updates
  lats_samp, res_samp, grad_norm_latent, grad_norm_res = langevin_update(lats_in, res_in)[0:4]

  # update metrics
  latent_grad_norm_metric.update_state(grad_norm_latent)
  res_grad_norm_metric.update_state(grad_norm_res)
  z_norm = tf.math.reduce_mean(tf.norm(tf.reshape(lats_samp, (lats_samp.shape[0], -1)), axis=1))
  z_norm_metric.update_state(z_norm)

  # perturb data with small gaussian noise
  images_data = images_data_in + config['data_epsilon'] * tf.random.normal(shape=tpu_image_size)
  # get generated images
  images_latent = gen(lats_samp, training=False)
  images_samp = images_latent + res_samp

  # update ebm
  loss = update_ebm(config, ebm, ebm_optim, images_data, images_samp, num_reps)
  # update loss record
  ebm_loss_metric.update_state(loss)

  return images_samp, images_latent

# training update function
def train_step(iterator):

  images_data_in = next(iterator)
  tuple_out = strategy.run(step_fn, args=(images_data_in,))

  return tuple_out


##################
# ## LEARNING ## #
##################

# records
ebm_loss_rec = np.zeros(shape=[config['num_training_steps']])
latent_grad_norm_rec = np.zeros(shape=[config['num_training_steps']])
res_grad_norm_rec = np.zeros(shape=[config['num_training_steps']])
z_norm_rec = np.zeros(shape=[config['num_training_steps']])
rec_names = ['EBM Loss', 'Latent Grad Norm', 'Res Grad Norm', 'Z Norm']

# start timer
time_check = time()

# training loop
print('Starting the training loop.')
for step in range(config['num_training_steps']):
  # training step on tf graph
  ims_samp_viz, ims_latent_viz = train_step(train_iterator)

  # update diagnostic records
  ebm_loss_rec[step] = float(ebm_loss_metric.result())
  latent_grad_norm_rec[step] = float(latent_grad_norm_metric.result())
  res_grad_norm_rec[step] = float(res_grad_norm_metric.result())
  z_norm_rec[step] = float(z_norm_metric.result())

  # print and plot diagnostics
  if step == 0 or (step + 1) % config['info_freq'] == 0:
    print('Train Step: {}/{}'.format(step + 1, config['num_training_steps']))
    print(
      'Energy Diff: {:.5f}   Grad Norm Latent: {:.5f}   Grad Norm Res: {:.5f}   Z Norm: {:.5f}'.
        format(
          ebm_loss_rec[step],
          latent_grad_norm_rec[step],
          res_grad_norm_rec[step],
          z_norm_rec[step]
        )
    )
    if step > 0:
      print('Time per Batch: {:.2f}'.format((time() - time_check) / config['info_freq']))
      time_check = time()

  if (step + 1) % config['plot_freq'] == 0:
    rec_list = [ebm_loss_rec, latent_grad_norm_rec, res_grad_norm_rec, z_norm_rec]
    plot_diagnostics(config, step, rec_list, rec_names)

  # save images and checkpoints
  if step == 0 or (step + 1) % config['log_freq'] == 0:
        save_model(config, step, strategy, ebm, ims_samp_viz, None, 
                   ims_latent_viz, ebm_optim, None, None, None)

  # reset metrics
  ebm_loss_metric.reset_states()
  res_grad_norm_metric.reset_states()
  latent_grad_norm_metric.reset_states()
  z_norm_metric.reset_states()
