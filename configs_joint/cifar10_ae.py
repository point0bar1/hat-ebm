####################
# ## CIFAR10 AE ## #
####################

# train gen and inf net using mse reconstruction loss
# gen net can be used as foundation of retrofit hat EBM

config={
  # paths for connecting to data (only one of the two will be used)
  "data_path": "/PATH/TO/DATA/",  # data path if not using gs bucket, else None
  "gs_path": None,  # name of gs bucket if used, else None

  # device type ("tpu" or "gpu" or "cpu")
  "device_type": "gpu",

  # experiment info
  "exp_name": "cifar10_ae",
  "exp_dir": "/PATH/TO/OUTPUT/",
  "num_epochs": 100,
  "epoch_steps_tr": 499,
  "epoch_steps_test": 99,
  "batch_size": 128,
  "image_dims": [32, 32, 3],
  "state_dims": [16, 16, 1],

  # data parameters
  "data_type": "cifar10",
  "data_epsilon": 2.0e-2,
  "random_crop": False,
  "test_split": "test",

  # generator network parameters
  "gen_type": "gen_decoder",
  "gen_factor": 2.0,
  # optimization parameters for gen
  "gen_optim_type": "adam",
  "gen_lr_info": [[1e-4, 0]],
  "gen_opt_decay": 0.9999,

  # inference network parameters
  "inf_type": "inf_encoder",
  # optimization parameters for inf
  "inf_optim_type": "adam",
  "inf_lr_info": [[1e-4, 0]],
  "inf_opt_decay": 0.9999,

  # loss parameters
  "loss_factor": 1000,

  # logging parameters
  "test_and_log_freq": 5,
  "save_to_cloud": False,  # True if gs_path is not None and want to save to cloud, else False
}
