import tensorflow_datasets as tfds
import os

# this code is based on the imagenet2012 download instructions from the article:
# https://towardsdatascience.com/preparing-the-imagenet-dataset-with-tensorflow-c681916014ee
# by Pascal Janetsky

# for imagenet2012, 'dataset_dir' should contain the files 
# ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar
# that can be downloaded from: https://www.image-net.org/

# for celeb_a, 'dataset_dir' should be the path to the extracted folder
# img_align_celeba 
# downloaded from the 'Align&Cropped Images' here: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset',
    default='imagenet2012',
    help='either "imagenet2012" or "celeb_a"'
)
parser.add_argument(
    '--dataset_dir',
    default='/PATH/TO/RAW_DATA/',
    help='directory where raw data is stored'
)
parser.add_argument(
    '--temp_dir',
    default='/PATH/TO/TEMP_DIR/',
    help='a temporary directory for saving data during processing'
)
parser.add_argument(
    '--save_dir',
    default='/PATH/TO/TF_RECORDS/',
    help='location to save tf records'
)
args = parser.parse_args()


dl_config = tfds.download.DownloadConfig(
    extract_dir=os.path.join(args.temp_dir, 'extracted'),
    manual_dir=args.dataset_dir
)

tfds.builder(args.dataset, data_dir=args.save_dir).download_and_prepare(download_config=dl_config)
