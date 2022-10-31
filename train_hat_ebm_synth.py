# train Hat EBM for image synthesis
# latent distribution is N(0, I), residuals are conditional on latent states

import os
from time import time
from datetime import datetime
import importlib

import tensorflow as tf

from utils import setup_exp, save_model, plot_ims, plot_diagnostics
from init import initialize_strategy, initialize_net_and_optim, determinism_test
from init import initialize_persistent
from data import get_dataset
from ebm_utils import make_langevin_update, update_ebm, update_gen

import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    'config_name', 
    default='configs_synth/cifar10_synth.py', 
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
    optim_decay=config['ebm_opt_decay'],
    weight_path=config['ebm_weights'],
    optim_weights=config['ebm_optim_weights'],
    optim_state_dims=config['image_dims']
  )
# test deterministic ouput
determinism_test(strategy, ebm, config['image_dims'])
# model summary
ebm.summary()

# load generator and inference net
gen, gen_optim = \
  initialize_net_and_optim(
    config, 
    strategy, 
    config['gen_type'],
    optim_type=config['gen_optim_type'],
    optim_lr_info=config['gen_lr_info'],
    optim_decay=config['gen_opt_decay'],
    weight_path=config['gen_weights'],
    optim_weights=config['gen_optim_weights'],
    optim_state_dims=config['state_dims']
  )
# test deterministic ouput
determinism_test(strategy, gen, config['state_dims'])
# model summary
gen.summary()


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

# bank of images and latents to update the generator
z_bank = initialize_persistent(
                            config, 
                            strategy, 
                            init_type='noise',
                            persistent_size=config['persistent_size'],
                            state_dims=config['state_dims'],
                            persistent_init_scale=config['init_scale_latent'],
                            persistent_path=config['persistent_z_path']
                          )
images_bank = initialize_persistent(
                            config, 
                            strategy, 
                            init_type='noise',
                            persistent_size=config['persistent_size'],
                            state_dims=config['image_dims'],
                            persistent_init_scale=config['init_scale_res'],
                            persistent_path=config['persistent_im_path'],
                            z_bank=z_bank,
                            gen=gen
                          )


with strategy.scope():
  # metrics
  ebm_loss_metric = tf.keras.metrics.Mean('ebm_loss', dtype=tf.float32)
  gen_loss_metric = tf.keras.metrics.Mean('gen_loss', dtype=tf.float32)
  res_grad_norm_metric = tf.keras.metrics.Mean('res_grad_norm', dtype=tf.float32)


###########################
# ## TF GRAPH BUILDERS ## #
###########################

# graph for langevin sampling
langevin_update = make_langevin_update(config, ebm, gen)

@tf.function
def step_fn(images_data_in, images_samp_bank, z_samp_bank):

  # initial samples for mcmc
  lats_in = config['init_scale_latent'] * tf.random.normal(shape=tpu_latent_size)
  res_in = config['init_scale_res'] * tf.random.normal(shape=tpu_image_size)

  # langevin updates
  lats_samp, res_samp, _, grad_norm_res = langevin_update(lats_in, res_in)[0:4]

  # update metrics
  res_grad_norm_metric.update_state(grad_norm_res)

  # perturb data with small gaussian noise
  images_data = images_data_in + config['data_epsilon'] * tf.random.normal(shape=tpu_image_size)
  # get generated images
  images_latent = gen(lats_samp, training=False)
  images_samp = images_latent + res_samp

  # update ebm
  loss = update_ebm(config, ebm, ebm_optim, images_data, images_samp, num_reps)
  # update loss record
  ebm_loss_metric.update_state(loss)

  # samples to update generator
  update_inds_shuffle = tf.random.shuffle(tf.range(0, tf.shape(images_samp_bank)[0]))
  update_inds_gen = update_inds_shuffle[0:per_replica_batch_size]
  images_update = tf.gather(images_samp_bank, update_inds_gen)
  z_update = tf.gather(z_samp_bank, update_inds_gen)
  # update generator
  loss_gen = update_gen(config, gen, gen_optim, z_update, images_update, num_reps)
  # update loss record
  gen_loss_metric.update_state(loss_gen)

  # update bank of viz states used to update the gen in future steps
  images_samp_bank = tf.tensor_scatter_nd_update(images_samp_bank,
      tf.reshape(update_inds_gen, shape=[-1, 1]), tf.identity(images_samp))
  # update bank of latents states used to update the gen in future steps
  z_samp_bank = tf.tensor_scatter_nd_update(z_samp_bank,
      tf.reshape(update_inds_gen, shape=[-1, 1]), tf.identity(lats_samp))

  return images_samp, images_latent, images_samp_bank, z_samp_bank

# training update function
def train_step(iterator, images_samp_bank, z_samp_bank):

  images_data_in = next(iterator)
  tuple_out = strategy.run(
                step_fn, 
                args=(
                  images_data_in,
                  images_samp_bank,
                  z_samp_bank
                )
              )

  return tuple_out


##################
# ## LEARNING ## #
##################

# records
ebm_loss_rec = np.zeros(shape=[config['num_training_steps']])
gen_loss_rec = np.zeros(shape=[config['num_training_steps']])
res_grad_norm_rec = np.zeros(shape=[config['num_training_steps']])
rec_names = ['EBM Loss', 'Gen Loss', 'Res Grad Norm']

# start timer
time_check = time()

# training loop
print('Starting the training loop.')
for step in range(config['start_training_step'], config['num_training_steps']):
  # training step on tf graph
  ims_samp_viz, ims_latent_viz, images_bank, z_bank = \
      train_step(train_iterator, images_bank, z_bank)

  # update diagnostic records
  ebm_loss_rec[step] = ebm_loss_metric.result().numpy()
  gen_loss_rec[step] = gen_loss_metric.result().numpy()
  res_grad_norm_rec[step] = res_grad_norm_metric.result().numpy()

  # print and plot diagnostics
  if step == 0 or (step + 1) % config['info_freq'] == 0:
    print('{} Train Step: {}/{}'.format(args.config_name, step + 1, config['num_training_steps']))
    print(
      'Energy Diff: {:.5f}   Gen Loss: {:.5f}   Grad Norm Res: {:.5f}'.
        format(
          ebm_loss_rec[step],
          gen_loss_rec[step],
          res_grad_norm_rec[step]
        )
    )
    if step > 0:
      print('Time per Batch: {:.2f}'.format((time() - time_check) / config['info_freq']))
      time_check = time()

  if (step + 1) % config['plot_freq'] == 0:
    plot_diagnostics(config, step, [ebm_loss_rec, gen_loss_rec, res_grad_norm_rec], rec_names)

  # save images and checkpoints
  if step == 0 or (step + 1) % config['log_freq'] == 0:
    save_model(config, step, strategy, ebm, ims_samp_viz, gen, ims_latent_viz, 
               ebm_optim, gen_optim, images_bank, z_bank)

  # reset metrics
  ebm_loss_metric.reset_states()
  gen_loss_metric.reset_states()
  res_grad_norm_metric.reset_states()
