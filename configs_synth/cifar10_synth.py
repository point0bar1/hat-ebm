####################
# ## PARAMETERS ## #
####################

# train cifar10 model for synthesis

config = {
  # paths for connecting to data (only one of the two will be used)
  "data_path": "/PATH/TO/DATA/",  # data path if not using gs bucket, else None
  "gs_path": None,  # name of gs bucket if used, else None

  # device type ("tpu" or "gpu" or "cpu")
  "device_type": "gpu",

  # experiment info
  "exp_name": "cifar10_synth",
  "exp_dir": "/PATH/TO/OUTPUT/",
  "start_training_step": 0,
  "num_training_steps": 50000,
  "batch_size": 128,
  "image_dims": [32, 32, 3],
  "state_dims": [128],

  # data parameters
  "data_type": "cifar10",
  "data_epsilon": 1e-3,
  "random_crop": False,

  # ebm network parameters
  "ebm_type": "ebm_sngan",
  # weights for restarting exp
  "ebm_weights": None,
  "ebm_optim_weights": None,
  # optimization parameters for ebm
  "ebm_optim_type": "adam",
  "ebm_lr_info": [[1e-4, 0]],
  "ebm_opt_decay": 0.9999,

  # generator network parameters
  "gen_type": "gen_sngan",
  "gen_factor": 2.0,
  # weights for restarting exp
  "gen_weights": None,
  "gen_optim_weights": None,
  # optimization parameters for generator
  "gen_optim_type": "adam",
  "gen_lr_info": [[1e-4, 0]],
  "gen_opt_decay": 0.9999,
  "gen_loss_factor_coop": 1000,

  # langevin sampling parameters
  "epsilon": 5e-4,
  "mcmc_steps": 50,
  "mcmc_temp": 1e-8,
  "use_noise": True,
  "update_latents": False,

  # scales for initializing latent and residual image states
  "init_scale_latent": 1.0,
  "init_scale_res": 0.0,
  # number of images/latents in persistent banks
  "persistent_size": 10000,
  # persistent weights for gen update
  "persistent_im_path": None,
  "persistent_z_path": None,

  # gradient clipping for langevin
  "max_langevin_norm_res": 2.5,
  "clip_langevin_grad_res": True,
  # gradient clipping for nets
  "max_grad_norm": 50.0,
  "clip_ebm_grad": False,

  # logging parameters
  "info_freq": 50,
  "plot_freq": 500,
  "log_freq": 1000,
  "save_to_cloud": False,  # True if gs_path is not None and want to save to cloud, else False
}
