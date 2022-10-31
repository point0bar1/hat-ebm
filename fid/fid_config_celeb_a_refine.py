####################
# ## PARAMETERS ## #
####################

config = {
  # paths for connecting to data (only one of the two will be used)
  "data_path": "/PATH/TO/DATA/",  # data path if not using gs bucket, else None
  "gs_path": None,  # name of gs bucket if used, else None

  # location to save results
  "exp_name": 'fid_out/celeb_a_refine_eval',
  "exp_dir": "/PATH/TO/OUTPUT/",
  # device type ('tpu' or 'gpu' or 'cpu')
  "device_type": 'gpu',

  # experiment dependent parameters
  "epsilon": 1e-4,
  "epsilon_latent": 5e-3,
  "mcmc_temp": 1e-6,
  "update_latents": True,  # for joint learning to True, for synth learning to False

  # data type
  "data_type": 'celeb_a',  # cifar10, celeb_a, imagenet2012
  "random_crop": False,
  "split": "train",

  # ebm network
  "ebm_type": 'ebm_sngan',  # we only use ebm_sngan
  "ebm_weights": "/PATH/TO/EBM_WEIGHTS.ckpt",
  # gen network
  "gen_type": "gen_sngan",  # gen_decoder for retrofit, otherwise gen_sngan
  "gen_weights": "/PATH/TO/GEN_WEIGHTS.ckpt",
  # net params
  "gen_factor": 1.0,  # 1.0 for refinement, 2.0 for all other exps
  "ebm_ngf": None,  # only used for imagenet exps (1024 for small net, 2048 for large net)
  "gen_ngf": None,  # only used for imagenet exps (1024 for small net, 2048 for large net)

  # exp params
  "num_fid_rounds": 520,
  "batch_size": 96,
  "image_dims": [64, 64, 3],  # cifar10: [32, 32, 3], celeb_a [64, 64, 3], imagenet: [128, 128, 3]
  "state_dims": [128],  # [16, 16, 1] for retrofit, otherwise [128]

  # initial scales of latent and res states
  "init_scale_latent": 1.0,
  "init_scale_res": 0.0,

  # sampling parameters
  # for synth exps, we recommend about 125 mcmc steps during testing (~2.5x the steps used in training)
  "mcmc_steps": 125,
  "use_noise": True,

  # clipping parameters for sampling
  "clip_langevin_grad": True,
  "max_langevin_norm": 2.5,
}
