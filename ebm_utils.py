import tensorflow as tf


# langevin update loop in latent and residual space
def make_langevin_update(config, ebm, gen):
  @tf.function
  def langevin_update(latents_in, res_in):

    # initial samples for visual check
    images_samp_init = tf.identity(gen(latents_in, training=False) + res_in)

    # container for grad diagnostic
    grad_norm_latent = tf.constant(0, dtype=tf.float32)
    grad_norm_res = tf.constant(0, dtype=tf.float32)

    latents_samp = tf.identity(latents_in)
    res_samp = tf.identity(res_in)

    # langevin updates
    for _ in tf.range(config['mcmc_steps']):
      if config['update_latents']:
        # update latents
        with tf.GradientTape() as tape:
          tape.watch(latents_samp)
          images_samp = gen(latents_samp, training=False)
          energy = ebm(images_samp + res_samp, training=False) / config['mcmc_temp']
          energy_sum = tf.math.reduce_sum(energy)

        # gradient for latent states
        grads = tape.gradient(energy_sum, latents_samp)

        # update latents with gradient
        latents_samp -= ((config['epsilon_latent'] ** 2) / 2) * grads
        if config['use_noise']:
          # update latents with noise term
          latents_samp += config['epsilon_latent'] * tf.random.normal(shape=tf.shape(latents_samp))

        # record gradient norm
        grad_norm_step = tf.norm(tf.reshape(grads, shape=[tf.shape(latents_samp)[0], -1]), axis=1)
        grad_norm_latent += ((config['epsilon_latent'] ** 2) / 2) * tf.math.reduce_mean(grad_norm_step)

      # update residual image
      with tf.GradientTape() as tape:
        tape.watch(res_samp)
        images_samp = gen(latents_samp, training=False)
        energy = ebm(images_samp + res_samp, training=False) / config['mcmc_temp']
        energy_sum = tf.math.reduce_sum(energy)
        # gaussian prior (optional)
        if 'tau_res'in config.keys() and config['tau_res'] > 0:
          energy_sum += config['tau_res'] * tf.math.reduce_sum(res_samp**2)

      # gradient for residual states
      grads = tape.gradient(energy_sum, res_samp)

      # clip gradient norm (set to large value that won't interfere with standard dynamics)
      if 'clip_langevin_grad_res' in config.keys() and config['clip_langevin_grad_res']:
        max_norm_scaled = config['max_langevin_norm_res'] / ((config['epsilon'] ** 2) / 2)
        grads = tf.clip_by_norm(grads, max_norm_scaled, axes=[1, 2, 3])

      # update residual images with gradient
      res_samp -= ((config['epsilon'] ** 2) / 2) * grads
      if config['use_noise']:
        # update residual images with noise term
        res_samp += config['epsilon'] * tf.random.normal(shape=tf.shape(res_samp))

      # record gradient norm
      grad_norm_step = tf.norm(tf.reshape(grads, shape=[tf.shape(res_samp)[0], -1]), axis=1)
      grad_norm_res += ((config['epsilon'] ** 2) / 2) * tf.math.reduce_mean(grad_norm_step)

    # final sampled states
    images_out = gen(latents_samp, training=False) + res_samp
    # get mean of metrics
    grad_norm_latent /= config['mcmc_steps']
    grad_norm_res /= config['mcmc_steps']

    return latents_samp, res_samp, grad_norm_latent, grad_norm_res, images_out, images_samp_init

  return langevin_update


# update energy network with data and synthesized images
def update_ebm(config, ebm, ebm_optim, images_data, images_samp, rep_scale):
  with tf.GradientTape() as tape:
    # energy of data and model samples
    en_pos = ebm(images_data, training=False)
    en_neg = ebm(images_samp, training=False)
    # maximum likelihood 'loss'
    loss = (tf.math.reduce_mean(en_pos) - tf.math.reduce_mean(en_neg)) / config['mcmc_temp']
    # rescale to adjust for summation over number of replicas
    loss_scaled = loss / rep_scale
  # get gradients
  ebm_grads = tape.gradient(loss_scaled, ebm.trainable_variables)
  # clip gradient norm
  if config['clip_ebm_grad']:
    ebm_grads = [tf.clip_by_norm(g, config['max_grad_norm']) for g in ebm_grads]
  # update ebm
  ebm_optim.apply_gradients(list(zip(ebm_grads, ebm.trainable_variables)))

  return loss


# update generator using pairs of latent samples and synthesized images
def update_gen(config, gen, gen_optim, z_update, images_update, rep_scale):
  with tf.GradientTape() as tape:
    # mse loss and scaling
    coop_error = (gen(z_update, training=False) - images_update) ** 2
    loss_gen = config['gen_loss_factor_coop'] * tf.math.reduce_mean(coop_error)
    # rescale to adjust for summation over number of replicas
    loss_gen_scaled = loss_gen / rep_scale

  # get gradients
  gen_grads = tape.gradient(loss_gen_scaled, gen.trainable_variables)

  # update gen and inf nets
  gen_optim.apply_gradients(list(zip(gen_grads, gen.trainable_variables)))

  return loss_gen
