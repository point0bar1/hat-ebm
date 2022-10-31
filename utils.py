import os
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import tensorflow as tf
import numpy as np
import pickle


def upload_blob(bucket_name, source_file_name, destination_blob_name):
  """Uploads a file to the bucket."""
  storage_client = storage.Client()
  bucket = storage_client.bucket(bucket_name)
  blob = bucket.blob(destination_blob_name)

  blob.upload_from_filename(source_file_name)

  print(
    "File {} uploaded to {}.".format(
      source_file_name, destination_blob_name
    )
  )

def download_blob(bucket_name, source_file_name, destination_blob_name):
  """Downloads a file from the bucket."""

  storage_client = storage.Client()
  bucket = storage_client.get_bucket(bucket_name)
  blob = bucket.blob(source_file_name)

  blob.download_to_filename(destination_blob_name)

  print(
    "File {} downloaded to {}.".format(
      source_file_name, destination_blob_name
    )
  )

# visualize images with pixels in range [-1, 1]
def plot_ims(path, ims): 
  if tf.is_tensor(ims):
    ims = ims.numpy()
  ims = (np.clip(ims, -1., 1.) + 1) / 2

  # dimensions of image grid
  nrows = int(np.ceil(ims.shape[0] ** 0.5))
  ncols = int(np.ceil(ims.shape[0] ** 0.5))

  fig = plt.figure(figsize=(nrows, ncols))
  grid = ImageGrid(
    fig, 111,  # similar to subplot(111)
    nrows_ncols=(nrows, ncols),
    axes_pad=0.05,  # pad between axes in inch.
  )

  grid[0].get_yaxis().set_ticks([])
  grid[0].get_xaxis().set_ticks([])

  for ax, im in zip(grid, ims.tolist()):
    im = np.array(im)
    if im.shape[2] == 1:
      im = np.tile(im, (1, 1, 3))
    ax.imshow(im)
    ax.axis("off")
  plt.savefig(path, format="pdf", dpi=2000)
  plt.close()

# save copy of code in the experiment folder
def save_code(exp_dir, code_file_list, gs_path=None, save_to_cloud=False):
  def save_file(file_name):
    file_in = open(file_name, 'r')
    file_out = open(os.path.join(exp_dir, 'code/', os.path.basename(file_name)), 'w')
    for line in file_in:
      file_out.write(line)
  for code_file in code_file_list:
    save_file(code_file)
    if gs_path is not None and save_to_cloud == True:
      upload_blob(gs_path,
                  os.path.join(exp_dir, 'code/', os.path.basename(code_file)), 
                  os.path.join(exp_dir, 'code/', os.path.basename(code_file))
      )

# make folders, save config and code
def setup_exp(exp_dir, folder_list, code_file_list=[], gs_path=None, save_to_cloud=False):
  # make directory for saving results
  if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
  for folder in ['code'] + folder_list:
    if not os.path.exists(os.path.join(exp_dir, folder)):
      os.mkdir(os.path.join(exp_dir, folder))
  save_code(exp_dir, code_file_list, gs_path, save_to_cloud)

# plot diagnostics for learning
def plot_diagnostics(config, step, records, record_names, fontsize=6):
  exp_folder = os.path.join(config['exp_dir'], config['exp_name'])

  # axis tick size
  matplotlib.rc('xtick', labelsize=6)
  matplotlib.rc('ytick', labelsize=6)
  fig = plt.figure()

  # make diagnostic plots
  for i, record in enumerate(records):
    record_name = record_names[i]

    # make figure
    ax = fig.add_subplot(len(records), 1, i+1)
    ax.plot(record[0:(step+1)])
    ax.set_title(record_name, fontsize=fontsize)
    ax.set_xlabel('batch', fontsize=fontsize)

    # save numpy files of record
    np.save(os.path.join(exp_folder, 'plots/'+record_name+'.npy'), record[0:(step+1)])
    if config['save_to_cloud']:
      upload_blob(config['gs_path'],
                  os.path.join(exp_folder, 'plots/'+record_name+'.npy'),
                  os.path.join(exp_folder, 'plots/'+record_name+'.npy'))

  # save figure
  plt.subplots_adjust(hspace=1.2, wspace=0.6)
  plt.savefig(os.path.join(exp_folder, 'plots', 'diagnosis_plot.png'))
  plt.close()
  if config['save_to_cloud']:
    upload_blob(config['gs_path'],
                os.path.join(exp_folder, 'plots', 'diagnosis_plot.png'),
                os.path.join(exp_folder, 'plots', 'diagnosis_plot.png'))

# function for saving model results
def save_model(config, step, strategy, ebm, ims_samp, gen, ims_latent,
               ebm_optim, gen_optim, ims_persistent, z_persistent):

  if config['gs_path'] is not None:
    from google.cloud import storage

  # folder for results
  exp_folder = os.path.join(config['exp_dir'], config['exp_name'])
  if config['gs_path'] is not None:
    # gs cloud location for saving nets
    net_folder = os.path.join('gs://'+config['gs_path'], exp_folder)
  else:
    # local folder for saving nets
    net_folder = exp_folder
  per_replica_batch_size = config['batch_size'] // strategy.num_replicas_in_sync

  # save example synthesized images
  ims_samp_viz = strategy.gather(ims_samp, axis=0)
  plot_ims(os.path.join(exp_folder, 'shortrun/shortrun'+str(step+1)+'.pdf'), 
           ims_samp_viz[0:per_replica_batch_size])
  if config['save_to_cloud']:
    upload_blob(config['gs_path'],
                os.path.join(exp_folder, 'shortrun/shortrun'+str(step+1)+'.pdf'),
                os.path.join(exp_folder, 'shortrun/shortrun'+str(step+1)+'.pdf'))

  # save example generator samples
  if ims_latent is not None:
    ims_latent_viz = strategy.gather(ims_latent, axis=0)
    plot_ims(os.path.join(exp_folder, 'shortrun/latent_viz'+str(step+1)+'.pdf'), 
             ims_latent_viz[0:per_replica_batch_size])
    if config['save_to_cloud']:
      upload_blob(config['gs_path'],
                  os.path.join(exp_folder, 'shortrun/latent_viz'+str(step+1)+'.pdf'),
                  os.path.join(exp_folder, 'shortrun/latent_viz'+str(step+1)+'.pdf'))

  if step > 0:
    if ebm is not None:
      # save ebm model (save to cloud if gs_path specified in config, otherwise local save)
      ebm.save_weights(os.path.join(net_folder, 'checkpoints/ebm_{}.ckpt'.format(step+1)))

    if gen is not None:
      # save gen model (save to cloud if gs_path specified in config, otherwise local save)
      gen.save_weights(os.path.join(net_folder, 'checkpoints/gen_{}.ckpt'.format(step+1)))

    # save optim
    if ebm_optim is not None:
      ebm_optim_weights = ebm_optim.get_weights()
      with open(os.path.join(exp_folder, 'checkpoints/ebm_optim_{}.ckpt'.format(step+1)), 'wb') as f:
        pickle.dump(ebm_optim_weights, f)
      if config['save_to_cloud']:
        upload_blob(config['gs_path'],
                    os.path.join(exp_folder, 'checkpoints/ebm_optim_'+str(step+1)+'.ckpt'),
                    os.path.join(exp_folder, 'checkpoints/ebm_optim_'+str(step+1)+'.ckpt'))
        # remove to save space
        os.remove(os.path.join(exp_folder, 'checkpoints/ebm_optim_'+str(step+1)+'.ckpt'))

    # save gen optim
    if gen_optim is not None:
      gen_optim_weights = gen_optim.get_weights()
      with open(os.path.join(exp_folder, 'checkpoints/gen_optim_{}.ckpt'.format(step+1)), 'wb') as f:
        pickle.dump(gen_optim_weights, f)
      if config['save_to_cloud']:
        upload_blob(config['gs_path'],
                    os.path.join(exp_folder, 'checkpoints/gen_optim_'+str(step+1)+'.ckpt'),
                    os.path.join(exp_folder, 'checkpoints/gen_optim_'+str(step+1)+'.ckpt'))
        # remove to save space
        os.remove(os.path.join(exp_folder, 'checkpoints/gen_optim_'+str(step+1)+'.ckpt'))

    # save persistent states to cloud (once every 20 checkpoints)
    if ims_persistent is not None and (step + 1) % (20 * config['log_freq']) == 0:
      ims_persistent_gather = strategy.gather(ims_persistent, axis=0)
      with open(os.path.join(exp_folder, 'checkpoints/persistent_'+str(step+1)+'.ckpt'), 'wb') as f:
        pickle.dump(ims_persistent_gather, f)
      if config['save_to_cloud']:
        upload_blob(config['gs_path'],
                    os.path.join(exp_folder, 'checkpoints/persistent_'+str(step+1)+'.ckpt'),
                    os.path.join(exp_folder, 'checkpoints/persistent_'+str(step+1)+'.ckpt'))
        # remove to save space
        os.remove(os.path.join(exp_folder, 'checkpoints/persistent_'+str(step+1)+'.ckpt'))

    # save persistent z to cloud (once every 20 checkpoints)
    if z_persistent is not None and (step + 1) % (20 * config['log_freq']) == 0:
      z_persistent_gather = strategy.gather(z_persistent, axis=0)
      with open(os.path.join(exp_folder, 'checkpoints/persistent_z_'+str(step+1)+'.ckpt'), 'wb') as f:
        pickle.dump(z_persistent_gather, f)
      if config['save_to_cloud']:
        upload_blob(config['gs_path'],
                    os.path.join(exp_folder, 'checkpoints/persistent_z_'+str(step+1)+'.ckpt'),
                    os.path.join(exp_folder, 'checkpoints/persistent_z_'+str(step+1)+'.ckpt'))
        # remove to save space
        os.remove(os.path.join(exp_folder, 'checkpoints/persistent_z_'+str(step+1)+'.ckpt'))
