#!/usr/bin/env python3
''' Calculates the Frechet Inception Distance (FID) to evalulate GANs.
The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.
The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.
'''

import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from pathlib import Path
import tensorflow.compat.v1 as tf
from scipy import linalg
import warnings
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

class InvalidFIDException(Exception):
    pass


###############################
# ## PATHS TO SAVED IMAGES ## #
###############################

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--path1',
    default='/PATH/TO/images1.npy',
    help='Name of config file.'
)
parser.add_argument(
    '--path2',
    default='/PATH/TO/images2.npy',
    help='Name of config file.'
)
args = parser.parse_args()


###########################
# ## ORIGINAL FID CODE ## #
###########################

# only change is removing name string "FID_Inception_Net"
# https://github.com/bioinf-jku/TTUR/issues/42

def create_inception_graph(pth):
    """Creates a graph from saved GraphDef file."""
    # Creates graph from saved graph_def.pb.
    with tf.io.gfile.GFile( pth, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString( f.read())
        _ = tf.import_graph_def( graph_def)
#-------------------------------------------------------------------------------


# code for handling inception net derived from
#   https://github.com/openai/improved-gan/blob/master/inception_score/model.py
def _get_inception_layer(sess):
    """Prepares inception net for batched usage and returns pool_3 layer. """
    layername = 'pool_3:0'
    pool3 = sess.graph.get_tensor_by_name(layername)
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims is not None:
              #shape = [s.value for s in shape] TF 1.x
              shape = [s for s in shape] #TF 2.x
              new_shape = []
              for j, s in enumerate(shape):
                if s == 1 and j == 0:
                  new_shape.append(None)
                else:
                  new_shape.append(s)
              o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
    return pool3
#-------------------------------------------------------------------------------


def get_activations(images, sess, batch_size=50, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 256.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = _get_inception_layer(sess)
    n_images = images.shape[0]
    if batch_size > n_images:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = n_images
    n_batches = n_images//batch_size # drops the last batch if < batch_size
    pred_arr = np.empty((n_batches * batch_size,2048))
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
        start = i*batch_size
        
        if start+batch_size < n_images:
            end = start+batch_size
        else:
            end = n_images

        batch = images[start:end]
        pred = sess.run(inception_layer, {'ExpandDims:0': batch})
        pred_arr[start:end] = pred.reshape(batch.shape[0],-1)
    if verbose:
        print(" done")
    return pred_arr
#-------------------------------------------------------------------------------


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
#-------------------------------------------------------------------------------


def check_or_download_inception(inception_path):
    ''' Checks if the path to the inception file is valid, or downloads
        the file if it is not present. '''
    INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    if inception_path is None:
        inception_path = '/tmp'
    inception_path = Path(inception_path)
    model_file = inception_path / 'classify_image_graph_def.pb'
    if not model_file.exists():
        print("Downloading Inception model")
        from urllib import request
        import tarfile
        fn, _ = request.urlretrieve(INCEPTION_URL)
        with tarfile.open(fn, mode='r') as f:
            f.extract('classify_image_graph_def.pb', str(model_file.parent))
    return str(model_file)
#-------------------------------------------------------------------------------


######################
# ## ADAPTED CODE ## #
######################

# we adapt the original FID code to be compatible with memory-efficient sequential numpy files

def calculate_activation_statistics(path, sess, verbose=False):

    act = np.zeros((0, 2048))

    # get activations across batches
    p = Path(path)
    with p.open('rb') as f:
        fsz = os.fstat(f.fileno()).st_size
        i = 0
        while f.tell() < fsz:
            if i == 0 or (i+1) % 50 == 0:
                print('Batch {}'.format(i+1))
            image_batch = np.load(f)
            act_batch = get_activations(image_batch, sess, image_batch.shape[0], verbose)
            act = np.concatenate((act, act_batch), 0)
            i += 1

    # get mean and covariance of activations and return
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_fid(path1, path2, inception_path):
    inception_path = check_or_download_inception(inception_path)

    with tf.device("/gpu:0"):
        create_inception_graph(str(inception_path))

    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        m1, s1 = calculate_activation_statistics(path1, sess)
        m2, s2 = calculate_activation_statistics(path2, sess)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value


# run experiment
fid_value = calculate_fid(args.path1, args.path2, None)
print("FID: ", fid_value)
