import tensorflow as tf

import os
from datetime import datetime
import importlib
import matplotlib.pyplot as plt
import numpy as np

from utils import setup_exp, plot_ims
from init import initialize_strategy, initialize_net_and_optim, determinism_test
from data import get_dataset

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    'config_name', 
    default='configs_joint/cifar10_ae.py', 
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
exp_folder = os.path.join(config['exp_dir'], config['exp_name'])

# setup folders, save code
setup_exp(os.path.join(config['exp_dir'], config['exp_name']), 
          ['checkpoints', 'plots', 'viz'], 
          [code_file for code_file in
            [
              os.path.basename(__file__), 
              'nets.py', 
              'utils.py', 
              'data.py', 
              'init.py', 
              args.config_name
            ]
          ],
          config['gs_path'], config['save_to_cloud'])

# initialize distribution strategy
strategy = initialize_strategy(config['device_type'])


##################################
# ## INITIALIZE NETS AND DATA ## #
##################################

# load generator and inference net
gen, gen_optim = \
  initialize_net_and_optim(
    config, 
    strategy, 
    config['gen_type'],
    optim_type=config['gen_optim_type'],
    optim_lr_info=config['gen_lr_info'],
    optim_decay=config['gen_opt_decay']
  )
# test deterministic ouput
determinism_test(strategy, gen, config['state_dims'])

inf, inf_optim = \
  initialize_net_and_optim(
    config, 
    strategy, 
    config['inf_type'],
    optim_type=config['inf_optim_type'],
    optim_lr_info=config['inf_lr_info'],
    optim_decay=config['inf_opt_decay']
  )
# test deterministic ouput
determinism_test(strategy, inf, config['image_dims'])

# tensor size for tpus
per_replica_batch_size = config['batch_size'] // strategy.num_replicas_in_sync
tpu_tensor_size = [per_replica_batch_size] + config['state_dims']

# initialize train data
train_dataset = strategy.distribute_datasets_from_function(
  lambda _: get_dataset(
    config['data_type'],
    per_replica_batch_size,
    config['image_dims'],
    'gs://' + config['gs_path'] if config['gs_path'] is not None else config['data_path'],
    random_crop=config['random_crop']
  )
)

test_dataset = strategy.distribute_datasets_from_function(
  lambda _: get_dataset(
    config['data_type'],
    per_replica_batch_size,
    config['image_dims'],
    'gs://' + config['gs_path'] if config['gs_path'] is not None else config['data_path'],
    random_crop=config['random_crop'],
    split=config['test_split']
  )
)


with strategy.scope():
  # records
  training_loss = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
  test_loss = tf.keras.metrics.Mean('testing_loss', dtype=tf.float32)


######################################
# ## TF GRAPHS FOR TRAIN AND TEST ## #
######################################

def get_loss(X_orig, X_recon):
  loss = tf.math.reduce_mean((X_orig - X_recon) ** 2)
  return config['loss_factor'] * loss

@tf.function
def train_epoch(train_loader_in):
  def train_step(X):
    with tf.GradientTape(persistent=True) as tape:
      latent_noise = config['data_epsilon'] * tf.random.normal(shape=tpu_tensor_size)
      X_reconstructed = gen(inf(X) + latent_noise, training=False)
      loss_ae = get_loss(X, X_reconstructed)

      # scale for summation across paralled devices
      loss_scaled = loss_ae / strategy.num_replicas_in_sync

    # get gradients
    gen_grads = tape.gradient(loss_scaled, gen.trainable_variables)
    inf_grads = tape.gradient(loss_scaled, inf.trainable_variables)
    del tape

    # update gen and inf nets
    gen_optim.apply_gradients(list(zip(gen_grads, gen.trainable_variables)))
    inf_optim.apply_gradients(list(zip(inf_grads, inf.trainable_variables)))

    training_loss.update_state(loss_ae)

  for _ in tf.range(config['epoch_steps_tr']):
    strategy.run(train_step, args=(next(train_loader_in),))

@tf.function
def test_epoch(test_loader_in):
  def test_step(X):
    X_reconstructed = gen(inf(X), training=False)
    loss_ae = get_loss(X, X_reconstructed)

    test_loss.update_state(loss_ae)

  for _ in tf.range(config['epoch_steps_test']):
    strategy.run(test_step, args=(next(test_loader_in),))

@tf.function
def viz_recon(X):
  latents = inf(X)
  return gen(latents, training=False), latents


#######################
# ## LEARNING LOOP # ##
#######################

# records for training info
train_loss_rec = np.zeros([config['num_epochs']])
test_loss_rec = np.zeros([config['num_epochs'] // config['test_and_log_freq']])

print('Training has begun.')
for epoch in range(config['num_epochs']):
  config['split'] = 'train'
  train_loader = iter(train_dataset)
  train_epoch(train_loader)
  print('Epoch {}: Training Loss={}'.format(epoch+1, round(float(training_loss.result()), 4)))

  # update training record then reset the metric objects
  train_loss_rec[epoch] = round(float(training_loss.result()), 4)
  training_loss.reset_states()

  if (epoch+1) % config['test_and_log_freq'] == 0:
    # evaluate test data
    config['split'] = 'test'
    test_loader = iter(test_dataset)
    test_epoch(test_loader)
    print('Epoch {}: Test Loss={}'.format(epoch+1, round(float(test_loss.result()), 4)))

    # update training record then reset the metric objects
    test_loss_rec[epoch // config['test_and_log_freq']] = round(float(test_loss.result()), 4)
    test_loss.reset_states()

    # save checkpoint and diagnostic plots
    test_num = (epoch + 1) // config['test_and_log_freq']
    plt.plot(np.arange(1, epoch+2), train_loss_rec[0:(epoch+1)])
    plt.plot(config['test_and_log_freq']*np.arange(1, test_num+1), test_loss_rec[0:test_num])
    plt.savefig(os.path.join(config['exp_dir'], config['exp_name'], 'plots', 'loss_fig.png'))
    plt.close()

    # visualize images
    ims_viz_data = next(test_loader)
    ims_viz_recon, ims_latent_recon = strategy.run(viz_recon, args=(ims_viz_data,))
    ims_viz_recon_gather = strategy.gather(ims_viz_recon, 0)
    # plot
    plot_ims(os.path.join(exp_folder, 'viz/recon'+str(epoch+1)+'.pdf'), ims_viz_recon_gather)

    # save network
    if config['save_to_cloud']:
      save_folder = os.path.join('gs://'+config['gs_path'], exp_folder)
    else:
      save_folder = exp_folder
    gen.save_weights(os.path.join(save_folder, 'checkpoints/gen_{}.ckpt'.format(epoch+1)))
    inf.save_weights(os.path.join(save_folder, 'checkpoints/inf_{}.ckpt'.format(epoch+1)))
