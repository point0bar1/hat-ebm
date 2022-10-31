####################
# ## PARAMETERS ## #
####################

# use pretrained generator as foundation of joint hat EBM

config = {
  # paths for connecting to data (only one of the two will be used)
  "data_path": "/PATH/TO/DATA/",  # data path if not using gs bucket, else None
  "gs_path": None,  # name of gs bucket if used, else None

  # device type ("tpu" or "gpu" or "cpu")
  "device_type": "gpu",

  # experiment info
  "exp_name": "celeb_a_refine",
  "exp_dir": "/PATH/TO/OUTPUT/",
  "num_training_steps": 25000,
  "batch_size": 128,
  "image_dims": [64, 64, 3],
  "state_dims": [128],

  # data parameters
  "data_type": "celeb_a",
  "data_epsilon": 1e-3,
  "random_crop": False,

  # ebm network parameters
  "ebm_type": "ebm_sngan",
  # optimization parameters for ebm
  "ebm_optim_type": "adam",
  "ebm_lr_info": [[1e-5, 0]],
  "ebm_opt_decay": 0.9999,

  # generator network parameters and weight path
  "gen_type": "gen_sngan",
  "gen_factor": 1.0,
  "gen_weights": "/PATH/TO/GEN_WEIGHTS.ckpt",

  # langevin sampling parameters
  "epsilon": 1e-4,
  "epsilon_latent": 5e-3,
  "mcmc_steps": 100,
  "mcmc_temp": 1e-6,
  "use_noise": True,
  "update_latents": True,

  # scales for initializing latent and residual image states
  "init_scale_latent": 1.0,
  "init_scale_res": 0.0,

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
