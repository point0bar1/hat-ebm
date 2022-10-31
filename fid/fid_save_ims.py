import os
from os.path import dirname, realpath
import sys
sys.path.append(dirname(dirname(realpath(__file__))))

from datetime import datetime
import importlib
from pathlib import Path

import numpy as np

import tensorflow as tf

from init import initialize_strategy, initialize_net_and_optim, determinism_test
from data import get_dataset
from utils import setup_exp, plot_ims
from ebm_utils import make_langevin_update

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--config_name', 
    default='fid/fid_config.py', 
    help='Name of config file.'
)
args = parser.parse_args()

# get experiment config
config_module = importlib.import_module(args.config_name.replace('/', '.')[:-3])
config = config_module.config

# give exp_name unique timestamp identifier
time_str = datetime.now().strftime('%y-%m-%d-%H-%M-%S')
config['exp_name'] = config['exp_name'] + '_' + time_str


##########################
# ## GENERATE SAMPLES ## #
##########################

# save images from tf2 model to npy files to use original fid code
def save_samples(strategy, config, ebm, gen=None, train_iterator=None, save_str='samples.pdf'):

  # location to save results
  exp_folder = os.path.join(config['exp_dir'], config['exp_name'])

  # tf graph for langevin updates
  langevin_update = make_langevin_update(config, ebm, gen)

  for i in range(config['num_fid_rounds']):
    print('Batch {} of {}'.format(i+1, config['num_fid_rounds']))

    # data images
    images_data = next(train_iterator)

    per_replica_batch_size = config['batch_size'] // strategy.num_replicas_in_sync
    # generate random latent samples on cpu
    z_init_cpu = tf.random.normal([config['batch_size']]+config['state_dims'])
    z_init_cpu *= config['init_scale_latent']
    def get_z_init(ctx):
      rep_id = ctx.replica_id_in_sync_group
      return z_init_cpu[(rep_id*per_replica_batch_size):((rep_id+1)*per_replica_batch_size)]
    z_init = strategy.experimental_distribute_values_from_function(get_z_init)
    # generate res samples on cpu
    res_init_cpu = tf.random.normal(shape=[config['batch_size']]+config['image_dims'])
    res_init_cpu *= config['init_scale_res']
    def get_res_init(ctx):
      rep_id = ctx.replica_id_in_sync_group
      return res_init_cpu[(rep_id*per_replica_batch_size):((rep_id+1)*per_replica_batch_size)]
    res_init = strategy.experimental_distribute_values_from_function(get_res_init)

    # run langevin updates to get sampled images
    images_sample, images_init = strategy.run(langevin_update, args=(z_init, res_init))[4:6]

    # record batch images
    p1 = Path(os.path.join(exp_folder, 'numpy_out/images1.npy'))
    with p1.open('ab') as f:
      images_data_numpy = np.clip(strategy.gather(images_data, 0).numpy(), -1, 1)
      images_data_uint8 = np.rint(255 * (images_data_numpy + 1) / 2).astype(np.uint8)
      np.save(f, images_data_uint8)
    p2 = Path(os.path.join(exp_folder, 'numpy_out/images2.npy'))
    with p2.open('ab') as f:
      images_sample_numpy = np.clip(strategy.gather(images_sample, 0).numpy(), -1, 1)
      images_sample_uint8 = np.rint(255 * (images_sample_numpy + 1) / 2).astype(np.uint8)
      np.save(f, images_sample_uint8)

    # visualize initial and final samples for first batch
    if i == 0:
      images_init_numpy = np.clip(strategy.gather(images_init, 0).numpy(), -1, 1)
      plot_ims(os.path.join(exp_folder, 'images/' + save_str), images_sample_numpy)
      plot_ims(os.path.join(exp_folder, 'images/init_' + save_str), images_init_numpy)
      plot_ims(os.path.join(exp_folder, 'images/data_' + save_str), images_data_numpy)


###############
# ## SETUP ## #
###############

# setup folders, save code, set seed and get device
setup_exp(os.path.join(config['exp_dir'], config['exp_name']), 
          ['images', 'numpy_out'],
          [
            code_file for code_file in 
              [
                'fid/' + os.path.basename(__file__),
                'nets.py',
                'utils.py',
                'data.py',
                'init.py',
                'ebm_utils.py',
                args.config_name
              ]
          ],
          None, save_to_cloud=False)

strategy = initialize_strategy(config['device_type'])


##################################################
# ## INITIALIZE NETS, DATA, PERSISTENT STATES ## #
##################################################

# load nets and optim
ebm, _ = \
  initialize_net_and_optim(
    config, 
    strategy, 
    config['ebm_type'],
    weight_path=config['ebm_weights']
  )
ebm.trainable = False

gen, _ = \
  initialize_net_and_optim(
    config, 
    strategy, 
    config['gen_type'],
    weight_path=config['gen_weights']
  )
gen.trainable = False

# test deterministic output of ebm
determinism_test(strategy, ebm, config['image_dims'])
# test deterministic ouput of generator
determinism_test(strategy, gen, config['state_dims'])

# initialize data
train_dataset = strategy.distribute_datasets_from_function(
  lambda _: get_dataset(
    config['data_type'],
    config['batch_size'] // strategy.num_replicas_in_sync,
    config['image_dims'],
    'gs://' + config['gs_path'] if config['gs_path'] is not None else config['data_path'],
    random_crop=config['random_crop'],
    split=config['split']
  )
)
train_iterator = iter(train_dataset)

# save bank of data and model samples as two np arrays
save_samples(strategy, config, ebm, gen, train_iterator)
